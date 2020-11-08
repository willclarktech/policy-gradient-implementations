import numpy as np  # type: ignore
import torch as T
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple

from core import Agent, Hyperparameters

Observation = List[float]


def calculate_return(rewards: List[float], gamma) -> float:
    return np.sum([reward * (gamma ** i) for i, reward in enumerate(rewards)])


def calculate_returns(rewards: List[float], gamma: float) -> List[float]:
    return [calculate_return(rewards[i:], gamma) for i, _ in enumerate(rewards)]


class ActorCritic(Agent):
    def __init__(self, hyperparameters: Hyperparameters,) -> None:
        super(ActorCritic, self).__init__()

        self.gamma = hyperparameters.gamma
        in_features = hyperparameters.env.observation_space.shape[0]
        num_actions = hyperparameters.env.action_space.n
        hidden_features = hyperparameters.hidden_features

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
            [*self.network.parameters(), *self.V.parameters()], lr=hyperparameters.alpha
        )

    def evaluate(self, observation: Observation) -> T.Tensor:
        output = self.network(self.process(observation))
        return self.V(output)

    def choose_action(self, observation: Observation) -> Tuple[int, T.Tensor]:
        output = self.network(self.process(observation))
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
