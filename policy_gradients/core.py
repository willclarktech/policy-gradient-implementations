import gym  # type: ignore
import numpy as np  # type: ignore
import torch as T
from typing import Any, List

from utils import plot_returns


class Hyperparameters:
    def __init__(
        self,
        env_name: str = "",
        n_episodes: int = 0,
        log_period: int = 1,
        hidden_features: List[int] = [],
        alpha: float = 1.0,
        gamma: float = 1.0,
        tau: float = 1.0,
        batch_size: int = 1,
        replay_buffer_capacity: int = 0,
    ) -> None:
        self.env = gym.make(env_name)

        self.n_episodes = n_episodes
        self.log_period = log_period

        self.hidden_features = hidden_features
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        self.batch_size = batch_size
        self.replay_buffer_capacity = replay_buffer_capacity


class Agent:
    def __init__(self):
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

    def process(self, observation: np.ndarray, dtype=T.float32) -> T.Tensor:
        return T.tensor(observation, dtype=dtype).to(self.device)

    def choose_action(self, *args) -> Any:
        pass

    def update(self, *args) -> None:
        pass
