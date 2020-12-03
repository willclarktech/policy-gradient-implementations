from typing import List, Tuple

from gym import spaces  # type: ignore
import numpy as np  # type: ignore
import torch as T
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from policy_gradients.core import BaseAgent, Hyperparameters
from policy_gradients.utils import mlp


def calculate_return(rewards: List[float], gamma: float) -> float:
    return np.sum([reward * (gamma ** i) for i, reward in enumerate(rewards)])


def calculate_returns(rewards: List[float], gamma: float) -> List[float]:
    return [calculate_return(rewards[i:], gamma) for i, _ in enumerate(rewards)]


class Agent(BaseAgent):
    # pylint: disable=invalid-name,not-callable
    def __init__(self, hyperparameters: Hyperparameters,) -> None:
        super().__init__(hyperparameters)
        self.gamma = hyperparameters.gamma

        env = hyperparameters.env
        if not isinstance(env.action_space, spaces.Discrete):
            raise ValueError("This agent only supports discrete action spaces")
        in_features = env.observation_space.shape[0]
        num_actions = env.action_space.n
        hidden_features = hyperparameters.hidden_features

        self.rewards: List[float]
        self.log_probabilities: T.Tensor

        self.policy = mlp([in_features, *hidden_features, num_actions], nn.ReLU).to(
            self.device
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=hyperparameters.alpha)

    def reset(self) -> None:
        self.rewards = []
        self.log_probabilities = T.tensor([]).to(self.device)

    def choose_action(self, observation: np.ndarray) -> Tuple[int, T.Tensor]:
        output = self.policy(self.process([observation]))
        probabilities = F.softmax(output, dim=1)
        distribution = distributions.Categorical(probs=probabilities)
        action = distribution.sample()
        log_probability = distribution.log_prob(action)
        return action.item(), log_probability

    def remember(self, log_probability: T.Tensor, reward: float) -> None:
        self.rewards.append(reward)
        self.log_probabilities = T.cat([self.log_probabilities, log_probability])

    def update(self) -> None:
        self.optimizer.zero_grad()
        G = T.tensor(calculate_returns(self.rewards, self.gamma)).to(self.device)
        loss = T.sum(-G * self.log_probabilities)
        loss.backward()
        self.optimizer.step()

    def train(self) -> None:
        self.policy.train()

    def eval(self) -> None:
        self.policy.eval()

    def load(self, load_dir: str) -> None:
        self.policy.load_state_dict(
            T.load(self.get_savefile_name(load_dir, "policy"), map_location=self.device)
        )

    def save(self, save_dir: str) -> None:
        T.save(self.policy.state_dict(), self.get_savefile_name(save_dir, "policy"))
