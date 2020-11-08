import numpy as np  # type: ignore
from typing import List, Tuple

Observation = List[float]
DigitisedObservation = List[int]


class Digitiser:
    def __init__(self):
        self.observation_space: List[DigitisedObservation] = []

    def get_observation_space(self) -> List[DigitisedObservation]:
        pass

    def digitise(self, observation: Observation) -> DigitisedObservation:
        pass


class CartPoleDigitiser(Digitiser):
    def __init__(
        self, bounds: Observation = [2.4, 4.0, 0.209, 4.0], n_boxes: float = 10
    ) -> None:
        super(CartPoleDigitiser, self).__init__
        self.cart_position_space = np.linspace(-bounds[0], bounds[0], n_boxes)
        self.cart_velocity_space = np.linspace(-bounds[1], bounds[1], n_boxes)
        self.pole_angle_space = np.linspace(-bounds[2], bounds[2], n_boxes)
        self.pole_velocity_space = np.linspace(-bounds[3], bounds[3], n_boxes)
        self.observation_space = self.get_observation_space()

    def get_observation_space(self) -> List[DigitisedObservation]:
        observations = []
        for i in range(len(self.cart_position_space) + 1):
            for j in range(len(self.cart_velocity_space) + 1):
                for k in range(len(self.pole_angle_space) + 1):
                    for l in range(len(self.pole_velocity_space) + 1):
                        observations.append([i, j, k, l])

        return observations

    def digitise(self, observation: Observation) -> DigitisedObservation:
        cart_position = np.digitize(observation[0], self.cart_position_space)
        cart_velocity = np.digitize(observation[1], self.cart_velocity_space)
        pole_angle = np.digitize(observation[2], self.pole_angle_space)
        pole_velocity = np.digitize(observation[3], self.pole_velocity_space)
        return [cart_position, cart_velocity, pole_angle, pole_velocity]
