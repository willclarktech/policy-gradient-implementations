import gym  # type: ignore
import numpy as np  # type: ignore
import torch as T
from typing import Any, List

from utils import plot_returns


class Hyperparameters:
    def __init__(
        self,
        env_name: str = "LunarLander-v2",
        n_episodes: int = 2000,
        log_period: int = 1,
        hidden_features: List[int] = [2048, 1536],
        alpha: float = 5e-6,
        gamma: float = 0.99,
    ) -> None:
        self.env = gym.make(env_name)

        self.n_episodes = n_episodes
        self.log_period = log_period

        self.hidden_features = hidden_features
        self.alpha = alpha
        self.gamma = gamma


class Agent:
    def __init__(self):
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

    def process(self, observation: np.ndarray, dtype=T.float32) -> T.Tensor:
        return T.tensor([observation], dtype=dtype).to(self.device)

    def choose_action(self, *args) -> Any:
        pass

    def update(self, *args) -> None:
        pass
