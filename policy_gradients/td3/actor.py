from typing import List

import torch as T
import torch.nn as nn
import torch.optim as optim

from policy_gradients.utils import mlp


class Actor(nn.Module):
    def __init__(
        self,
        in_features: int,
        action_dims: int,
        hidden_features: List[int],
        alpha: float,
    ) -> None:
        super().__init__()
        self.network = mlp(
            [in_features, *hidden_features, action_dims], nn.ReLU, nn.Tanh
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, observation: T.Tensor) -> T.Tensor:
        return self.network(observation)

    def save(self, filepath: str) -> None:
        T.save(self.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self.load_state_dict(T.load(filepath))
