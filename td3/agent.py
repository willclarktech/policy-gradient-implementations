import gym  # type: ignore
import numpy as np  # type: ignore
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from actor import Actor
from critic import Critic
from replay_buffer import Action, Observation, ReplayBuffer
from utils import update_target_network


class Agent:
    def __init__(
        self,
        env: gym.Env,
        batch_size: int = 100,
        replay_buffer_capacity: int = 1_000_000,
        gamma: float = 0.99,
        tau: float = 5e-3,
        noise: float = 0.2,
        noise_clip: float = 0.5,
        d: int = 2,
    ) -> None:
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        self.min_action = env.action_space.low
        self.max_action = env.action_space.high
        in_features = env.observation_space.shape[0]
        action_dims = env.action_space.shape[0]

        self.gamma = T.scalar_tensor(gamma).to(self.device)
        self.tau = tau
        self.noise = noise
        self.noise_clip = noise_clip
        self.d = d
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(
            replay_buffer_capacity, (in_features,), action_dims
        )

        self.critic_1 = Critic(in_features, action_dims).to(self.device)
        self.critic_2 = Critic(in_features, action_dims).to(self.device)
        self.actor = Actor(in_features, action_dims).to(self.device)
        self.critic_1_target = Critic(in_features, action_dims).to(self.device)
        self.critic_2_target = Critic(in_features, action_dims).to(self.device)
        self.actor_target = Actor(in_features, action_dims).to(self.device)
        self.update_target_networks(tau=1)

    def update_target_networks(self, tau: float) -> None:
        update_target_network(self.critic_1_target, self.critic_1, tau)
        update_target_network(self.critic_2_target, self.critic_2, tau)
        update_target_network(self.actor_target, self.actor, tau)

    def process(self, arr: np.ndarray, dtype=T.float32) -> T.Tensor:
        return T.tensor(arr, dtype=dtype).to(self.device)

    def choose_action(self, observation: Observation) -> Action:
        self.actor.eval()
        with T.no_grad():
            raw_action = (
                self.actor(self.process([observation])).flatten().detach().cpu().numpy()
            )
            noisy_action = raw_action + np.random.normal(0, self.noise, raw_action.size)
            return noisy_action.clip(self.min_action, self.max_action)

    def remember(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        observation_: Observation,
        done: bool,
    ) -> None:
        self.replay_buffer.store_transition(
            observation, action, reward, observation_, done
        )

    def learn(self, step: int) -> None:
        if self.replay_buffer.size < self.batch_size:
            return

        observation, action, reward, observation_, done = [
            self.process(t) for t in self.replay_buffer.sample(self.batch_size)
        ]

        self.critic_1.train()
        self.critic_2.train()
        with T.no_grad():
            raw_target_action = self.actor_target(observation_)
            target_action_noise = T.tensor(
                np.random.normal(0, self.noise, raw_target_action.shape),
                dtype=T.float32,
            ).clamp(-self.noise_clip, self.noise_clip)
            noisy_target_action = raw_target_action + target_action_noise
            target_action = noisy_target_action.clamp(
                self.min_action[0], self.max_action[0]
            )
            target_1_value_ = self.critic_1_target(observation_, target_action)
            target_2_value_ = self.critic_2_target(observation_, target_action)
            target_min_value_ = T.minimum(target_1_value_, target_2_value_)
            y = reward + target_min_value_ * self.gamma * (1 - done)

        for critic in [self.critic_1, self.critic_2]:
            critic.optimizer.zero_grad()
            q = critic(observation, action)
            critic_loss = F.mse_loss(q, y)
            critic_loss.backward()
            critic.optimizer.step()

        if step % self.d == 0:
            self.critic_1.eval()
            self.actor.train()
            self.actor.optimizer.zero_grad()
            action_current = self.actor(observation)
            actor_loss = -self.critic_1(observation, action_current).mean()
            actor_loss.backward()
            self.actor.optimizer.step()

            self.update_target_networks(self.tau)

    def save(self, save_directory: str) -> None:
        self.actor.save(f"{save_directory}/actor.zip")
        self.critic_1.save(f"{save_directory}/critic_1.zip")
        self.critic_2.save(f"{save_directory}/critic_2.zip")

    def load(self, save_directory: str) -> None:
        self.actor.load(f"{save_directory}/actor.zip")
        self.critic_1.load(f"{save_directory}/critic_1.zip")
        self.critic_2.load(f"{save_directory}/critic_2.zip")

        self.actor_target.load(f"{save_directory}/actor.zip")
        self.critic_1_target.load(f"{save_directory}/critic_1.zip")
        self.critic_2_target.load(f"{save_directory}/critic_2.zip")
