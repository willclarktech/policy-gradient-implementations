from typing import Optional, Tuple

import numpy as np  # type: ignore

Sample = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class ReplayBuffer:
    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        capacity: int,
        observation_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        seed: Optional[int] = None,
    ) -> None:
        self.capacity = capacity
        self.observations = np.zeros((capacity, *observation_shape))
        self.actions = np.zeros((capacity, *action_shape))
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.observations_ = np.zeros((capacity, *observation_shape))
        self.dones = np.zeros((capacity, 1), dtype=np.bool)
        self.rng = np.random.default_rng(seed)
        self.index = 0

    @property
    def size(self) -> int:
        return min(self.capacity, self.index)

    def store_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        observation_: np.ndarray,
        done: bool,
    ) -> None:
        i = self.index % self.capacity

        self.observations[i] = observation
        self.actions[i] = action
        self.rewards[i] = [reward]
        self.observations_[i] = observation_
        self.dones[i] = [done]

        self.index += 1

    def sample(self, batch_size: int = 1, deterministic: bool = False) -> Sample:
        n_samples = min(batch_size, self.size)
        indices = (
            range(batch_size)
            if deterministic
            else self.rng.integers(0, self.size, n_samples)
        )
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.observations_[indices],
            self.dones[indices],
        )
