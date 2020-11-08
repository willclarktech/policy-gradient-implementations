import numpy as np  # type: ignore
from typing import Dict, List, Tuple

Observation = Tuple[int, int, bool]


class Agent:
    def __init__(self, gamma: float = 0.99, epsilon: float = 0.001) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q: Dict[Tuple[Observation, int], float] = {}
        self.pi: Dict[Observation, List[float]] = {}

        self.sum_space = list(range(4, 22))
        self.dealer_space = list(range(1, 11))
        self.ace_space = [False, True]
        self.action_space: List[int] = [0, 1]  # [stick, hit]
        self.num_actions = len(self.action_space)

        self.observation_space: List[Observation] = []
        self.returns: Dict[Tuple[Observation, int], List[float]] = {}
        self.pairs_visited: Dict[
            Tuple[Observation, int], bool
        ] = {}  # first visit or not
        self.memory: List[Tuple[Observation, int, int]] = []

        self.init_values()

    def init_values(self) -> None:
        for total in self.sum_space:
            for card in self.dealer_space:
                for ace in self.ace_space:
                    observation = (total, card, ace)
                    self.observation_space.append(observation)
                    self.pi[observation] = [
                        1 / self.num_actions for _ in self.action_space
                    ]
                    for action in self.action_space:
                        self.Q[(observation, action)] = 0
                        self.returns[(observation, action)] = []
                        self.pairs_visited[(observation, action)] = False

    def reset(self) -> None:
        self.memory = []
        for observation in self.observation_space:
            for action in self.action_space:
                self.pairs_visited[(observation, action)] = False

    def policy(self, observation: Observation) -> int:
        action: int = np.random.choice(self.action_space, p=self.pi[observation])
        return action

    def remember(self, observation: Observation, action: int, reward: int) -> None:
        self.memory.append((observation, action, reward))

    def update_Q(self) -> None:
        for i, (observation, action, _) in enumerate(self.memory):
            G = 0.0
            if not self.pairs_visited[(observation, action)]:
                self.pairs_visited[(observation, action)] = True
                discount = 1.0
                for t, (_, _, reward) in enumerate(self.memory[i:]):
                    G += reward * discount
                    discount *= self.gamma

                self.returns[(observation, action)].append(G)

            self.Q[(observation, action)] = np.mean(self.returns[(observation, action)])

            self.update_pi(observation)

    def update_pi(self, observation: Observation) -> None:
        actions = [self.Q[(observation, action)] for action in self.action_space]
        A_star = np.argmax(actions)
        self.pi[observation] = [
            (1 - self.epsilon + self.epsilon / self.num_actions)
            if i == A_star
            else self.epsilon / self.num_actions
            for i in self.action_space
        ]
