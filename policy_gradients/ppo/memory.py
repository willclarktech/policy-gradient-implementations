from typing import List, Tuple

import numpy as np  # type: ignore
import torch as T


class PPOMemory:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.observations: List[np.ndarray] = []
        self.actions: List[T.Tensor] = []
        self.log_probabilities: List[T.Tensor] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    # pylint: disable=too-many-arguments
    def store(
        self,
        observation: np.ndarray,
        action: T.Tensor,
        log_probability: T.Tensor,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.log_probabilities.append(log_probability)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample(
        self,
    ) -> Tuple[
        T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor, List[np.ndarray]
    ]:
        n_observations = len(self.observations)
        batch_start_indices = np.arange(0, n_observations, self.batch_size)
        indices = np.arange(n_observations, dtype=np.int64)
        np.random.shuffle(indices)
        batch_indices = [indices[i : i + self.batch_size] for i in batch_start_indices]
        # pylint: disable=not-callable
        return (
            T.tensor(self.observations, dtype=T.float32),
            T.tensor(self.actions, dtype=T.float32),
            T.tensor(self.log_probabilities, dtype=T.float32),
            T.tensor(self.values, dtype=T.float32),
            T.tensor(self.rewards, dtype=T.float32),
            T.tensor(self.dones, dtype=T.int64),
            batch_indices,
        )
