from typing import List

import torch as T
import torch.nn as nn
import torch.optim as optim

from policy_gradients.utils import mlp


class Value(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: List[int], alpha: float,
    ) -> None:
        super().__init__()
        self.network = mlp([in_features, *hidden_features, 1], nn.ReLU)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, observation: T.Tensor) -> T.Tensor:
        return self.network(observation)
