import numpy as np  # type: ignore
import torch as T
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple

from core import BaseAgent, Hyperparameters


def calculate_return(rewards: List[float], gamma: float) -> float:
    return np.sum([reward * (gamma ** i) for i, reward in enumerate(rewards)])


def calculate_returns(rewards: List[float], gamma: float) -> List[float]:
    return [calculate_return(rewards[i:], gamma) for i, _ in enumerate(rewards)]


class Agent(BaseAgent):
    def __init__(self, hyperparameters: Hyperparameters,) -> None:
        super(Agent, self).__init__()

        self.gamma = hyperparameters.gamma
        in_features = hyperparameters.env.observation_space.shape[0]
        num_actions = hyperparameters.env.action_space.n
        hidden_features = hyperparameters.hidden_features

        self.rewards: List[float]
        self.log_probabilities: T.Tensor

        self.policy = nn.Sequential(
            nn.Linear(in_features, hidden_features[0]),
            nn.ReLU(),
            nn.Linear(hidden_features[0], hidden_features[1]),
            nn.ReLU(),
            nn.Linear(hidden_features[1], num_actions),
        ).to(self.device)
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
