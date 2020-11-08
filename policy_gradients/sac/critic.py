import torch as T
import torch.nn as nn
import torch.optim as optim
from typing import List


class Critic(nn.Module):
    def __init__(
        self,
        in_features: int,
        action_dims: int,
        hidden_features: List[int],
        alpha: float,
    ) -> None:
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features + action_dims, hidden_features[0]),
            nn.ReLU(),
            nn.Linear(hidden_features[0], hidden_features[1]),
            nn.ReLU(),
            nn.Linear(hidden_features[1], 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, observation: T.Tensor, action: T.Tensor) -> T.Tensor:
        inp = T.cat([observation, action], dim=1)
        return self.network(inp)
