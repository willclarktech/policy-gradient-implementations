import torch as T
import torch.nn as nn
import torch.optim as optim


class Value(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features_1: int = 256,
        hidden_features_2: int = 256,
        alpha: float = 3e-4,
    ) -> None:
        super(Value, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_features_1),
            nn.ReLU(),
            nn.Linear(hidden_features_1, hidden_features_2),
            nn.ReLU(),
            nn.Linear(hidden_features_2, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, observation: T.Tensor) -> T.Tensor:
        return self.network(observation)
