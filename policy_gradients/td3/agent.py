from gym import spaces  # type: ignore
import numpy as np  # type: ignore
import torch as T
import torch.nn.functional as F

from policy_gradients.core import BaseAgent, Hyperparameters
from policy_gradients.replay_buffer import ReplayBuffer
from policy_gradients.utils import update_target_network

from policy_gradients.td3.actor import Actor
from policy_gradients.td3.critic import Critic


class Agent(BaseAgent):
    # pylint: disable=invalid-name,not-callable,too-many-arguments,too-many-instance-attributes,too-many-locals
    def __init__(self, hyperparameters: Hyperparameters) -> None:
        super().__init__(hyperparameters)

        env = hyperparameters.env
        if not isinstance(env.action_space, spaces.Box):
            raise ValueError("This agent only supports box action spaces")
        self.min_action = env.action_space.low
        self.max_action = env.action_space.high
        in_dims = env.observation_space.shape
        action_dims = env.action_space.shape
        hidden_features = hyperparameters.hidden_features

        alpha = hyperparameters.alpha
        self.gamma = T.scalar_tensor(hyperparameters.gamma).to(self.device)
        self.tau = hyperparameters.tau
        self.noise = hyperparameters.noise
        self.noise_clip = hyperparameters.noise_clip
        self.d = hyperparameters.d
        self.batch_size = hyperparameters.batch_size
        self.replay_buffer = ReplayBuffer(
            hyperparameters.replay_buffer_capacity,
            in_dims,
            action_dims,
            hyperparameters.seed,
        )

        self.critic_1 = Critic(in_dims[0], action_dims[0], hidden_features, alpha).to(
            self.device
        )
        self.critic_2 = Critic(in_dims[0], action_dims[0], hidden_features, alpha).to(
            self.device
        )
        self.actor = Actor(in_dims[0], action_dims[0], hidden_features, alpha).to(
            self.device
        )
        self.critic_1_target = Critic(
            in_dims[0], action_dims[0], hidden_features, alpha
        ).to(self.device)
        self.critic_2_target = Critic(
            in_dims[0], action_dims[0], hidden_features, alpha
        ).to(self.device)
        self.actor_target = Actor(
            in_dims[0], action_dims[0], hidden_features, alpha
        ).to(self.device)
        self.update_target_networks(tau=1)
        self.rng = np.random.default_rng(hyperparameters.seed)

    def update_target_networks(self, tau: float) -> None:
        update_target_network(self.critic_1_target, self.critic_1, tau)
        update_target_network(self.critic_2_target, self.critic_2, tau)
        update_target_network(self.actor_target, self.actor, tau)

    def choose_action(self, observation: np.ndarray) -> np.ndarray:
        self.actor.eval()
        with T.no_grad():
            raw_action = (
                self.actor(self.process([observation])).flatten().detach().cpu().numpy()
            )
            noisy_action = raw_action + self.rng.normal(0, self.noise, raw_action.size)
            return noisy_action.clip(self.min_action, self.max_action)

    def remember(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        observation_: np.ndarray,
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
            target_action_noise = (
                T.tensor(
                    self.rng.normal(0, self.noise, raw_target_action.shape),
                    dtype=T.float32,
                )
                .clamp(-self.noise_clip, self.noise_clip)
                .to(self.device)
            )
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

    def train(self) -> None:
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()

    def load(self, load_dir: str) -> None:
        actor_state_dict = T.load(
            self.get_savefile_name(load_dir, "actor"), map_location=self.device
        )
        critic_1_state_dict = T.load(
            self.get_savefile_name(load_dir, "critic_1"), map_location=self.device
        )
        critic_2_state_dict = T.load(
            self.get_savefile_name(load_dir, "critic_2"), map_location=self.device
        )
        self.actor.load_state_dict(actor_state_dict)
        self.critic_1.load_state_dict(critic_1_state_dict)
        self.critic_2.load_state_dict(critic_2_state_dict)

        self.actor_target.load_state_dict(actor_state_dict)
        self.critic_1_target.load_state_dict(critic_1_state_dict)
        self.critic_2_target.load_state_dict(critic_2_state_dict)

    def save(self, save_dir: str) -> None:
        super().save(save_dir)
        T.save(self.actor.state_dict(), self.get_savefile_name(save_dir, "actor"))
        T.save(self.critic_1.state_dict(), self.get_savefile_name(save_dir, "critic_1"))
        T.save(self.critic_2.state_dict(), self.get_savefile_name(save_dir, "critic_2"))
