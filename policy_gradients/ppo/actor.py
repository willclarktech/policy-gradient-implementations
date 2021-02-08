from typing import List, Tuple, Union

import numpy as np  # type: ignore
import torch as T
from torch import distributions
from torch import nn
from torch import optim

from policy_gradients.utils import mlp


class Actor(nn.Module):
    # pylint: disable=invalid-name
    def __init__(
        self,
        in_features: int,
        action_dims: int,
        hidden_features: List[int],
        alpha: float,
    ):
        super().__init__()
        self.shared = mlp([in_features, *hidden_features], nn.ReLU, nn.ReLU)
        self.mu = nn.Linear(hidden_features[-1], action_dims)
        self.sigma = nn.Linear(hidden_features[-1], action_dims)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(
        self, observation: Union[np.ndarray, T.Tensor]
    ) -> distributions.normal.Normal:
        shared_output = self.shared(observation)
        means = self.mu(shared_output)
        stds = self.sigma(shared_output).clamp(1e-6, 1)
        return distributions.normal.Normal(means, stds)

    def sample(self, observation: np.ndarray) -> Tuple[T.Tensor, T.Tensor]:
        distribution = self.forward(observation)
        action = distribution.sample()
        log_probability = distribution.log_prob(action)
        return action, log_probability
