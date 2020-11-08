import numpy as np  # type: ignore
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from core import BaseAgent, Hyperparameters
from replay_buffer import ReplayBuffer

from ddpg.actor import Actor
from ddpg.critic import Critic
from ddpg.noise import OrnsteinUhlenbeckNoise


def update_target_network(
    target_network: nn.Module, online_network: nn.Module, tau: float
) -> None:
    for target_param, online_param in zip(
        target_network.parameters(), online_network.parameters()
    ):
        target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)
    target_network.eval()


class Agent(BaseAgent):
    def __init__(self, hyperparameters: Hyperparameters) -> None:
        super(Agent, self).__init__()
        self.gamma = T.scalar_tensor(hyperparameters.gamma).to(self.device)
        self.tau = hyperparameters.tau
        self.batch_size = hyperparameters.batch_size

        in_dims = hyperparameters.env.observation_space.shape
        action_dims = hyperparameters.env.action_space.shape
        hidden_features = hyperparameters.hidden_features

        self.replay_buffer = ReplayBuffer(
            hyperparameters.replay_buffer_capacity, in_dims, action_dims
        )
        mu = np.zeros(action_dims[0])
        self.noise = OrnsteinUhlenbeckNoise(mu)

        alpha = hyperparameters.alpha
        beta = hyperparameters.beta
        l2_weight_decay = hyperparameters.l2_weight_decay
        self.critic = Critic(
            in_dims[0], action_dims[0], hidden_features, beta, l2_weight_decay
        ).to(self.device)
        self.actor = Actor(in_dims[0], action_dims[0], hidden_features, alpha).to(
            self.device
        )
        self.critic_target = Critic(
            in_dims[0], action_dims[0], hidden_features, beta, l2_weight_decay
        ).to(self.device)
        self.actor_target = Actor(
            in_dims[0], action_dims[0], hidden_features, alpha
        ).to(self.device)
        self.update_target_networks(tau=1)

    def update_target_networks(self, tau: float) -> None:
        update_target_network(self.critic_target, self.critic, tau)
        update_target_network(self.actor_target, self.actor, tau)

    def reset(self) -> None:
        self.noise.reset()

    def choose_action(self, observation: np.ndarray) -> np.ndarray:
        self.actor.eval()
        with T.no_grad():
            mu = self.actor(self.process([observation]))
            return mu.flatten().detach().cpu().numpy() + self.noise()

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

    def learn(self) -> None:
        if self.replay_buffer.size < self.batch_size:
            return

        observation, action, reward, observation_, done = [
            self.process(t) for t in self.replay_buffer.sample(self.batch_size)
        ]

        self.critic.train()
        with T.no_grad():
            target_action = self.actor_target(observation_)
            target_value_ = self.critic_target(observation_, target_action)
            y = reward + target_value_ * self.gamma * (1 - done)

        q = self.critic(observation, action)
        critic_loss = F.mse_loss(q, y)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.train()
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
