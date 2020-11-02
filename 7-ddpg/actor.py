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
        hidden_features_1: int = 400,
        hidden_features_2: int = 300,
        alpha: float = 1e-4,
    ) -> None:
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(in_features, hidden_features_1)
        self.fc2 = nn.Linear(hidden_features_1, hidden_features_2)
        self.fc3 = nn.Linear(hidden_features_2, action_dims)

        self.network = nn.Sequential(
            self.fc1,
            nn.LayerNorm(hidden_features_1),
            nn.ReLU(),
            self.fc2,
            nn.LayerNorm(hidden_features_2),
            nn.ReLU(),
            self.fc3,
            nn.Tanh(),
        )

        self.initialize_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def initialize_weights(self) -> None:
        for layer in [self.fc1, self.fc2]:
            bound = 1.0 / np.sqrt(layer.in_features)
            init.uniform_(layer.weight, -bound, bound)
            init.uniform_(layer.bias, -bound, bound)

        out_bound = 3e-3
        init.uniform_(self.fc3.weight, -out_bound, out_bound)
        init.uniform_(self.fc3.bias, -out_bound, out_bound)

    def forward(self, observation: T.Tensor) -> T.Tensor:
        return self.network(observation)

    def save(self, filepath: str) -> None:
        T.save(self.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self.load_state_dict(T.load(filepath))
