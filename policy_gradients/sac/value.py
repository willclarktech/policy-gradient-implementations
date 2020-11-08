import torch as T
import torch.nn as nn
import torch.optim as optim
from typing import List


class Value(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: List[int], alpha: float,
    ) -> None:
        super(Value, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_features[0]),
            nn.ReLU(),
            nn.Linear(hidden_features[0], hidden_features[1]),
            nn.ReLU(),
            nn.Linear(hidden_features[1], 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, observation: T.Tensor) -> T.Tensor:
        return self.network(observation)
