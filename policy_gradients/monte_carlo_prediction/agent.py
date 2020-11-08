import numpy as np  # type: ignore
from typing import Dict, List, Tuple

Observation = Tuple[int, int, bool]


class Agent:
    def __init__(self, gamma: float = 0.99) -> None:
        self.gamma = gamma
        self.V: Dict[Observation, float] = {}

        self.sum_space = list(range(4, 22))
        self.dealer_space = list(range(1, 11))
        self.ace_space = [False, True]
        self.action_space = [0, 1]  # [stick, hit]

        self.observation_space: List[Observation] = []
        self.returns: Dict[Observation, List[float]] = {}
        self.observations_visited: Dict[Observation, bool] = {}  # first visit or not
        self.memory: List[Tuple[Observation, int]] = []

        self.init_values()

    def init_values(self) -> None:
        for total in self.sum_space:
            for card in self.dealer_space:
                for ace in self.ace_space:
                    observation = (total, card, ace)
                    self.observation_space.append(observation)
                    self.V[observation] = 0
                    self.returns[observation] = []
                    self.observations_visited[observation] = False

    def reset(self) -> None:
        self.memory = []
        for observation in self.observation_space:
            self.observations_visited[observation] = False

    def policy(self, observation: Observation) -> int:
        total, _, _ = observation
        action = 0 if total >= 20 else 1
        return action

    def remember(self, observation: Observation, reward: int) -> None:
        self.memory.append((observation, reward))

    def update_V(self) -> None:
        for i, (observation, _) in enumerate(self.memory):
            G = 0.0
            if not self.observations_visited[observation]:
                self.observations_visited[observation] = True
                discount = 1.0
                for t, (_, reward) in enumerate(self.memory[i:]):
                    G += reward * discount
                    discount *= self.gamma

                self.returns[observation].append(G)

        for observation, _ in self.memory:
            self.V[observation] = np.mean(self.returns[observation])
