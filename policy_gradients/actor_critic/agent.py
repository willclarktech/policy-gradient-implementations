from typing import List, Tuple

from gym import spaces  # type: ignore
import numpy as np  # type: ignore
import torch as T
import torch.distributions as distributions
import torch.nn as nn
import torch.optim as optim

from policy_gradients.core import BaseAgent, Hyperparameters
from policy_gradients.utils import mlp


def calculate_return(rewards: List[float], gamma: float) -> float:
    return np.sum([reward * (gamma ** i) for i, reward in enumerate(rewards)])


def calculate_returns(rewards: List[float], gamma: float) -> List[float]:
    return [calculate_return(rewards[i:], gamma) for i, _ in enumerate(rewards)]


class Agent(BaseAgent):
    # pylint: disable=invalid-name,too-many-arguments
    def __init__(
        self,
        hyperparameters: Hyperparameters,
    ) -> None:
        super().__init__(hyperparameters)
        self.gamma = hyperparameters.gamma

        env = hyperparameters.env
        if not isinstance(env.action_space, spaces.Discrete):
            raise ValueError("This agent only supports discrete action spaces")
        in_features = env.observation_space.shape[0]
        num_actions = env.action_space.n
        hidden_features = hyperparameters.hidden_features

        self.network = mlp([in_features, *hidden_features], nn.ReLU, nn.ReLU).to(
            self.device
        )
        self.V = nn.Linear(hidden_features[-1], 1).to(self.device)
        self.pi = nn.Sequential(
            nn.Linear(hidden_features[-1], num_actions),
            nn.Softmax(dim=-1),
        ).to(self.device)
        self.optimizer = optim.Adam(
            [*self.network.parameters(), *self.V.parameters()], lr=hyperparameters.alpha
        )

    def evaluate(self, observation: np.ndarray) -> T.Tensor:
        output = self.network(self.process([observation]))
        return self.V(output)

    def choose_action(self, observation: np.ndarray) -> Tuple[int, T.Tensor]:
        output = self.network(self.process([observation]))
        probabilities = self.pi(output)
        distribution = distributions.Categorical(probs=probabilities)
        action = distribution.sample()
        log_probability = distribution.log_prob(action)
        return action.item(), log_probability

    def learn(
        self,
        observation: np.ndarray,
        log_probability: T.Tensor,
        reward: float,
        done: bool,
        observation_: np.ndarray,
    ) -> None:
        self.optimizer.zero_grad()

        V = self.evaluate(observation)
        V_ = T.scalar_tensor(0) if done else self.evaluate(observation_)
        delta = reward + self.gamma * V_ - V

        actor_loss = -log_probability * delta
        critic_loss = delta ** 2
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()

    def train(self) -> None:
        self.network.train()
        self.V.train()
        self.pi.train()

    def eval(self) -> None:
        self.network.eval()
        self.V.eval()
        self.pi.eval()

    def load(self, load_dir: str) -> None:
        self.network.load_state_dict(
            T.load(
                self.get_savefile_name(load_dir, "network"), map_location=self.device
            )
        )
        self.V.load_state_dict(
            T.load(self.get_savefile_name(load_dir, "V"), map_location=self.device)
        )
        self.pi.load_state_dict(
            T.load(self.get_savefile_name(load_dir, "pi"), map_location=self.device)
        )

    def save(self, save_dir: str) -> None:
        super().save(save_dir)
        T.save(self.network.state_dict(), self.get_savefile_name(save_dir, "network"))
        T.save(self.V.state_dict(), self.get_savefile_name(save_dir, "V"))
        T.save(self.pi.state_dict(), self.get_savefile_name(save_dir, "pi"))
