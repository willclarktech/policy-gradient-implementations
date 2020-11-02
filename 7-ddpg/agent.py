import numpy as np  # type: ignore
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from typing import Dict, List

from actor import Actor
from critic import Critic
from noise import OrnsteinUhlenbeckNoise
from replay_buffer import Action, Observation, ReplayBuffer


def update_target_network(
    target_network: nn.Module, online_network: nn.Module, tau: float
) -> None:
    new_state_dict = {
        key: (
            (1 - tau) * param.clone() + tau * online_network.state_dict()[key].clone()
        )
        for key, param in target_network.state_dict().items()
    }
    target_network.load_state_dict(new_state_dict)


class Agent:
    def __init__(
        self,
        in_features: int,
        action_dims: int,
        batch_size: int = 64,
        replay_buffer_capacity: int = 1_000_000,
        gamma: float = 0.99,
        tau: float = 1e-3,
    ) -> None:
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.gamma = T.scalar_tensor(gamma).to(self.device)
        self.tau = tau
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer(
            replay_buffer_capacity, (in_features,), action_dims
        )
        mu = np.zeros(action_dims)
        self.noise = OrnsteinUhlenbeckNoise(mu)

        self.critic = Critic(in_features, action_dims).to(self.device)
        self.actor = Actor(in_features, action_dims).to(self.device)
        self.critic_target = Critic(in_features, action_dims).to(self.device)
        self.actor_target = Actor(in_features, action_dims).to(self.device)
        self.update_target_networks(tau=1)

    def update_target_networks(self, tau: float) -> None:
        update_target_network(self.critic_target, self.critic, tau)
        update_target_network(self.actor_target, self.actor, tau)

    def reset(self) -> None:
        self.noise.reset()

    def process(self, arr: np.ndarray, dtype=T.float32) -> T.Tensor:
        return T.tensor(arr, dtype=dtype).to(self.device)

    def choose_action(self, observation: Observation) -> Action:
        with T.no_grad():
            mu = self.actor(self.process([observation]))
            return mu.flatten().detach().cpu().numpy() + self.noise()

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

    def learn(self) -> None:
        if self.replay_buffer.size < self.batch_size:
            return

        observation, action, reward, observation_, done = [
            self.process(t) for t in self.replay_buffer.sample(self.batch_size)
        ]

        with T.no_grad():
            target_action = self.actor_target(observation_)
            target_value_ = self.critic_target(observation_, target_action)
            y = reward + target_value_ * self.gamma * (1 - done)

        q = self.critic(observation, action)
        critic_loss = F.mse_loss(q, y)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        action_current = self.actor(observation)
        actor_loss = -self.critic(observation, action_current).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_target_networks(self.tau)

    def save(self, save_directory: str) -> None:
        self.actor.save(f"{save_directory}/actor.zip")
        self.critic.save(f"{save_directory}/critic.zip")
        self.actor_target.save(f"{save_directory}/actor_target.zip")
        self.critic_target.save(f"{save_directory}/critic_target.zip")

    def load(self, save_directory: str) -> None:
        self.actor.load(f"{save_directory}/actor.zip")
        self.critic.load(f"{save_directory}/critic.zip")
        self.actor_target.load(f"{save_directory}/actor_target.zip")
        self.critic_target.load(f"{save_directory}/critic_target.zip")
