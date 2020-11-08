import numpy as np  # type: ignore
import torch as T
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from typing import List


class Critic(nn.Module):
    def __init__(
        self,
        in_features: int,
        action_dims: int,
        hidden_features: List[int],
        beta: float,
        l2_weight_decay: float,
    ) -> None:
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(in_features, hidden_features[0], False)
        self.fc2 = nn.Linear(hidden_features[0], hidden_features[1], False)
        self.fc3 = nn.Linear(hidden_features[1], 1)

        self.observation_value = nn.Sequential(
            self.fc1,
            nn.BatchNorm1d(hidden_features[0]),
            # nn.LayerNorm(hidden_features[0]),
            nn.ReLU(),
            self.fc2,
            nn.BatchNorm1d(hidden_features[1]),
            # nn.LayerNorm(hidden_features[1]),
        )
        self.action_value = nn.Linear(action_dims, hidden_features[1])
        self.q = nn.Sequential(nn.ReLU(), self.fc3)

        self.initialize_weights()
        self.optimizer = optim.Adam(
            self.parameters(), lr=beta, weight_decay=l2_weight_decay
        )

    def initialize_weights(self) -> None:
        for layer in [self.fc1, self.fc2, self.action_value]:
            bound = 1.0 / np.sqrt(layer.in_features)
            init.uniform_(layer.weight, -bound, bound)
            # init.uniform_(layer.bias, -bound, bound)

        out_bound = 3e-3
        init.uniform_(self.fc3.weight, -out_bound, out_bound)
        init.uniform_(self.fc3.bias, -out_bound, out_bound)

    def forward(self, observation: T.Tensor, action: T.Tensor) -> T.Tensor:
        observation_value = self.observation_value(observation)
        action_value = self.action_value(action)
        # NOTE: Another option would be to concat the values instead of adding them
        return self.q(observation_value.add(action_value))

    def save(self, filepath: str) -> None:
        T.save(self.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self.load_state_dict(T.load(filepath))