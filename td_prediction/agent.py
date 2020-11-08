import numpy as np  # type: ignore
from typing import Dict, List, Tuple

Observation = Tuple[float, float, float, float]


class Agent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.99) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.V: Dict[float, float] = {}

        self.action_space = [0, 1]  # [left, right]
        self.observation_space: List[float] = []

        self.init_values()

    def init_values(self) -> None:
        max_theta = 0.2094
        self.observation_space = np.linspace(-max_theta, max_theta, 10)
        for i in range(len(self.observation_space) + 1):
            self.V[i] = 0

    def process_observation(self, observation: Observation) -> int:
        return np.digitize(observation[2], self.observation_space)

    def policy(self, observation: Observation) -> int:
        theta = self.process_observation(observation)
        # Move left if the pole is to the left, move right otherwise
        return int(theta >= 6)

    def update_V(
        self, observation: Observation, reward: int, observation_: Observation
    ) -> None:
        theta = self.process_observation(observation)
        theta_ = self.process_observation(observation_)
        self.V[theta] += self.alpha * (
            reward + (self.gamma * self.V[theta_]) - self.V[theta]
        )
