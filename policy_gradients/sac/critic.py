from typing import List

import torch as T
import torch.nn as nn
import torch.optim as optim

from policy_gradients.utils import mlp


class Critic(nn.Module):
    def __init__(
        self,
        in_features: int,
        action_dims: int,
        hidden_features: List[int],
        alpha: float,
    ) -> None:
        super().__init__()
        self.network = mlp([in_features + action_dims, *hidden_features, 1], nn.ReLU)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, observation: T.Tensor, action: T.Tensor) -> T.Tensor:
        inp = T.cat([observation, action], dim=1)
        return self.network(inp)
