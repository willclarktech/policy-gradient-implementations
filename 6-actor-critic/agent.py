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
        hidden_features: List[int] = [2048, 1536],
        alpha: float = 5e-6,
        gamma: float = 0.99,
    ) -> None:
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.gamma = gamma

        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_features[0]),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(hidden_features[i], hidden_features[i + 1]), nn.ReLU()
                )
                for i, _ in enumerate(hidden_features[:-1])
            ],
        ).to(self.device)
        self.V = nn.Linear(hidden_features[-1], 1)
        self.pi = nn.Sequential(
            nn.Linear(hidden_features[-1], num_actions), nn.Softmax(dim=-1),
        )
        self.optimizer = optim.Adam(
            [*self.network.parameters(), *self.V.parameters()], lr=alpha
        )

    def process_observation(self, observation: Observation) -> T.Tensor:
        return T.tensor([observation], dtype=T.float32).to(self.device)

    def evaluate(self, observation: Observation) -> T.Tensor:
        processed_observation = self.process_observation(observation)
        output = self.network(processed_observation)
        return self.V(output)

    def choose_action(self, observation: Observation) -> Tuple[int, T.Tensor]:
        processed_observation = self.process_observation(observation)
        output = self.network(processed_observation)
        probabilities = self.pi(output)
        distribution = distributions.Categorical(probs=probabilities)
        action = distribution.sample()
        log_probability = distribution.log_prob(action)
        return action.item(), log_probability

    def update(
        self,
        observation: Observation,
        log_probability: T.Tensor,
        reward: float,
        done: bool,
        observation_: Observation,
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
