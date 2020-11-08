import torch as T
import torch.distributions as distributions
import torch.nn as nn
import torch.optim as optim
from typing import Tuple


class Actor(nn.Module):
    def __init__(
        self,
        in_features: int,
        action_dims: int,
        hidden_features_1: int = 256,
        hidden_features_2: int = 256,
        alpha: float = 3e-4,
        epsilon: float = 1e-6,
    ) -> None:
        super(Actor, self).__init__()
        self.epsilon = epsilon
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_features_1),
            nn.ReLU(),
            nn.Linear(hidden_features_1, hidden_features_2),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_features_2, action_dims)
        self.sigma = nn.Linear(hidden_features_2, action_dims)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, observation: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        output = self.shared(observation)
        return (
            self.mu(output),
            self.sigma(output).clamp(self.epsilon, 1),
        )

    def sample(
        self, observation: T.Tensor, reparameterize=True
    ) -> Tuple[T.Tensor, T.Tensor]:
        means, stds = self.forward(observation)
        distribution = distributions.normal.Normal(means, stds)
        sample = distribution.rsample() if reparameterize else distribution.sample()
        action = T.tanh(sample)
        log_probability = distribution.log_prob(sample) - (
            (1 - action.pow(2)) + self.epsilon
        ).log().sum(
            dim=1, keepdim=True
        )  # TODO: sum after subtraction?
        return action, log_probability
