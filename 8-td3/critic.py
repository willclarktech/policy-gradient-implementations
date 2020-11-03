import numpy as np  # type: ignore
import torch as T
import torch.nn as nn
import torch.optim as optim


class Critic(nn.Module):
    def __init__(
        self,
        in_features: int,
        action_dims: int,
        hidden_features_1: int = 256,
        hidden_features_2: int = 256,
        alpha: float = 1e-3,
    ) -> None:
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features + action_dims, hidden_features_1),
            nn.ReLU(),
            nn.Linear(hidden_features_1, hidden_features_2),
            nn.ReLU(),
            nn.Linear(hidden_features_2, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, observation: T.Tensor, action: T.Tensor) -> T.Tensor:
        inp = T.cat([observation, action], dim=1)
        return self.network(inp)

    def save(self, filepath: str) -> None:
        T.save(self.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self.load_state_dict(T.load(filepath))
