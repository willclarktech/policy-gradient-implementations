from typing import List, Tuple

import torch as T
import torch.distributions as distributions
import torch.nn as nn
import torch.optim as optim

from policy_gradients.utils import mlp


class Actor(nn.Module):
    # pylint: disable=invalid-name,too-many-arguments
    def __init__(
        self,
        in_features: int,
        action_dims: int,
        hidden_features: List[int],
        alpha: float,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.shared = mlp([in_features, *hidden_features], nn.ReLU, nn.ReLU)
        self.mu = nn.Linear(hidden_features[-1], action_dims)
        self.sigma = nn.Linear(hidden_features[-1], action_dims)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, observation: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        output = self.shared(observation)
        return (
            self.mu(output),
            self.sigma(output).clamp(self.epsilon, 1),
        )

    def sample(
        self, observation: T.Tensor, reparameterize: bool = True
    ) -> Tuple[T.Tensor, T.Tensor]:
        means, stds = self.forward(observation)
        distribution = distributions.normal.Normal(means, stds)
        sample = distribution.rsample() if reparameterize else distribution.sample()
        action = T.tanh(sample)
        log_probability = distribution.log_prob(sample) - (
            (1 - action.pow(2)) + self.epsilon
        ).log().sum(dim=1, keepdim=True)
        return action, log_probability
