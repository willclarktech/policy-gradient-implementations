import numpy as np


class Agent:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.V = {}

        self.sum_space = list(range(4, 22))
        self.dealer_space = list(range(1, 11))
        self.ace_space = [False, True]
        self.action_space = [0, 1]  # [stick, hit]

        self.observation_space = []
        self.returns = {}
        self.observations_visited = {}  # first visit or not
        self.memory = []

        self.init_values()

    def init_values(self):
        for total in self.sum_space:
            for card in self.dealer_space:
                for ace in self.ace_space:
                    observation = (total, card, ace)
                    self.observation_space.append(observation)
                    self.V[observation] = 0
                    self.returns[observation] = []
                    self.observations_visited[observation] = False

    def reset(self):
        self.memory = []
        for observation in self.observation_space:
            self.observations_visited[observation] = False

    def policy(self, observation):
        total, _, _ = observation
        action = 0 if total >= 20 else 1
        return action

    def remember(self, observation, reward):
        self.memory.append((observation, reward))

    def update_V(self):
        for i, (observation, _) in enumerate(self.memory):
            G = 0
            if not self.observations_visited[observation]:
                self.observations_visited[observation] = True
                discount = 1
                for t, (_, reward) in enumerate(self.memory[i:]):
                    G += reward * discount
                    discount *= self.gamma

                self.returns[observation].append(G)

        for observation, _ in self.memory:
            self.V[observation] = np.mean(self.returns[observation])
