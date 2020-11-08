import torch as T
import torch.nn as nn
import torch.optim as optim
from typing import List


class Actor(nn.Module):
    def __init__(
        self,
        in_features: int,
        action_dims: int,
        hidden_features: List[int],
        alpha: float,
    ) -> None:
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_features[0]),
            nn.ReLU(),
            nn.Linear(hidden_features[0], hidden_features[1]),
            nn.ReLU(),
            nn.Linear(hidden_features[1], action_dims),
            nn.Tanh(),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, observation: T.Tensor) -> T.Tensor:
        return self.network(observation)

    def save(self, filepath: str) -> None:
        T.save(self.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self.load_state_dict(T.load(filepath))
