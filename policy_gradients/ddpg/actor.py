from typing import List

import numpy as np  # type: ignore
import torch as T
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim


class Actor(nn.Module):
    def __init__(
        self,
        in_features: int,
        action_dims: int,
        hidden_features: List[int],
        alpha: float,
    ) -> None:
        super().__init__()

        features = [in_features, *hidden_features, action_dims]
        self.fcs = [
            nn.Linear(features[i], features[i + 1]) for i, _ in enumerate(features[:-1])
        ]

        # NOTE: The DDPG paper uses BatchNorm but PyTorch seems to have some problems
        self.network = nn.Sequential(
            *[
                nn.Sequential(fc, nn.LayerNorm(hidden_features[i]), nn.ReLU(),)
                for i, fc in enumerate(self.fcs[:-1])
            ],
            nn.Sequential(self.fcs[-1], nn.Tanh(),)
        )

        self.initialize_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def initialize_weights(self) -> None:
        for layer in self.fcs[:-1]:
            bound = 1.0 / np.sqrt(layer.in_features)
            init.uniform_(layer.weight, -bound, bound)
            init.uniform_(layer.bias, -bound, bound)

        out_bound = 3e-3
        init.uniform_(self.fcs[-1].weight, -out_bound, out_bound)
        init.uniform_(self.fcs[-1].bias, -out_bound, out_bound)

    def forward(self, observation: T.Tensor) -> T.Tensor:
        return self.network(observation)

    def save(self, filepath: str) -> None:
        T.save(self.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self.load_state_dict(T.load(filepath))
