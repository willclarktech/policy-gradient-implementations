from gym import spaces  # type: ignore
import numpy as np  # type: ignore
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from policy_gradients.core import BaseAgent, Hyperparameters
from policy_gradients.replay_buffer import ReplayBuffer

from policy_gradients.ddpg.actor import Actor
from policy_gradients.ddpg.critic import Critic
from policy_gradients.ddpg.noise import OrnsteinUhlenbeckNoise


def update_target_network(
    target_network: nn.Module, online_network: nn.Module, tau: float
) -> None:
    for target_param, online_param in zip(
        target_network.parameters(), online_network.parameters()
    ):
        target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)
    target_network.eval()


class Agent(BaseAgent):
    # pylint: disable=invalid-name,too-many-arguments,too-many-instance-attributes
    def __init__(self, hyperparameters: Hyperparameters) -> None:
        super().__init__(hyperparameters)
        self.gamma = T.scalar_tensor(hyperparameters.gamma).to(self.device)
        self.tau = hyperparameters.tau
        self.batch_size = hyperparameters.batch_size

        env = hyperparameters.env
        if not isinstance(env.action_space, spaces.Box):
            raise ValueError("This agent only supports box action spaces")
        self.min_action = env.action_space.low
        self.max_action = env.action_space.high
        in_dims = env.observation_space.shape
        action_dims = env.action_space.shape
        hidden_features = hyperparameters.hidden_features

        self.replay_buffer = ReplayBuffer(
            hyperparameters.replay_buffer_capacity,
            in_dims,
            action_dims,
            hyperparameters.seed,
        )
        mu = np.zeros(action_dims[0])
        self.noise = OrnsteinUhlenbeckNoise(mu, seed=hyperparameters.seed)

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
            noisy_action = mu.flatten().detach().cpu().numpy() + self.noise()
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

    def train(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic.eval()

    def load(self, load_dir: str) -> None:
        actor_state_dict = T.load(
            self.get_savefile_name(load_dir, "actor"), map_location=self.device
        )
        critic_state_dict = T.load(
            self.get_savefile_name(load_dir, "critic"), map_location=self.device
        )
        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)
        self.actor_target.load_state_dict(actor_state_dict)
        self.critic_target.load_state_dict(critic_state_dict)

    def save(self, save_dir: str) -> None:
        super().save(save_dir)
        T.save(self.actor.state_dict(), self.get_savefile_name(save_dir, "actor"))
        T.save(self.critic.state_dict(), self.get_savefile_name(save_dir, "critic"))
