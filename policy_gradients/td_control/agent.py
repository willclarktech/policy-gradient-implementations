import numpy as np  # type: ignore
from typing import Dict, List, Tuple

from td_control.digitiser import (
    DigitisedObservation,
    Digitiser,
    Observation,
)


class Agent:
    def __init__(
        self,
        n_actions: int,
        digitiser: Digitiser,
        alpha: float = 0.01,
        gamma: float = 0.99,
        epsilon_max: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_step: float = 0.01,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_step = epsilon_step

        self.Q: Dict[Tuple[Tuple, int], float] = {}

        self.action_space = list(range(n_actions))
        self.digitiser = digitiser

        self.init_Q()

    def init_Q(self) -> None:
        for observation in self.digitiser.observation_space:
            for action in self.action_space:
                self.Q[(tuple(observation), action)] = 0.0

    def process_observation(self, observation: Observation) -> Tuple:
        return tuple(self.digitiser.digitise(observation))

    def policy(self, observation: Observation) -> int:
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)

        processed_observation = self.process_observation(observation)
        actions = [
            self.Q[(processed_observation, action)] for action in self.action_space
        ]
        return np.argmax(actions)

    def update_Q(
        self,
        observation: Observation,
        action: int,
        reward: int,
        observation_: Observation,
    ) -> None:
        processed_observation = self.process_observation(observation)
        processed_observation_ = self.process_observation(observation_)
        q_ = np.max(
            [self.Q[(processed_observation_, action)] for action in self.action_space]
        )
        self.Q[(processed_observation, action)] += self.alpha * (
            reward + (self.gamma * q_) - self.Q[(processed_observation, action)]
        )

    def decrement_epsilon(self) -> None:
        self.epsilon = np.max([self.epsilon - self.epsilon_step, self.epsilon_min])
