import numpy as np  # type: ignore
import torch as T
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple

Observation = List[float]


def calculate_return(rewards: List[float], gamma) -> float:
    return np.sum([reward * (gamma ** i) for i, reward in enumerate(rewards)])


def calculate_returns(rewards: List[float], gamma: float) -> List[float]:
    return [calculate_return(rewards[i:], gamma) for i, _ in enumerate(rewards)]


class Agent:
    def __init__(
        self,
        in_features: int,
        num_actions: int,
        hidden_features: int = 128,
        alpha: float = 0.0005,
        gamma: float = 0.99,
    ) -> None:
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.rewards: List[float]
        self.log_probabilities: T.Tensor

        self.policy = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, num_actions),
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=alpha)

    def reset(self) -> None:
        self.rewards = []
        self.log_probabilities = T.tensor([])

    def process_observation(self, observation: Observation) -> T.Tensor:
        return T.tensor([observation], dtype=T.float32).to(self.device)

    def choose_action(self, observation: Observation) -> Tuple[int, T.Tensor]:
        processed_observation = self.process_observation(observation)
        output = self.policy(processed_observation)
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
