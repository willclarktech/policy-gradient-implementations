from typing import List, Optional, Tuple

import numpy as np  # type: ignore
import torch as T


class PPOMemory:
    # pylint: disable=too-many-instance-attributes
    def __init__(self, batch_size: int, seed: Optional[int] = None):
        self.batch_size = batch_size
        self.observations: List[np.ndarray] = []
        self.actions: List[T.Tensor] = []
        self.log_probabilities: List[T.Tensor] = []
        self.values: List[float] = []
        self.next_values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.rng = np.random.default_rng(seed)

    # pylint: disable=too-many-arguments
    def store(
        self,
        observation: np.ndarray,
        action: T.Tensor,
        log_probability: T.Tensor,
        value: float,
        reward: float,
        done: bool,
        value_next: float,
    ) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.log_probabilities.append(log_probability)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_values.append(value_next)

    def sample(
        self,
    ) -> Tuple[
        T.Tensor,
        T.Tensor,
        T.Tensor,
        T.Tensor,
        T.Tensor,
        T.Tensor,
        T.Tensor,
        List[np.ndarray],
    ]:
        n_examples = len(self.rewards)
        batch_start_indices = np.arange(0, n_examples, self.batch_size)
        indices = np.arange(n_examples, dtype=np.int64)
        self.rng.shuffle(indices)
        batch_indices = [indices[i : i + self.batch_size] for i in batch_start_indices]
        # pylint: disable=not-callable
        return (
            T.tensor(self.observations, dtype=T.float32),
            T.tensor(self.actions, dtype=T.float32),
            T.tensor(self.log_probabilities, dtype=T.float32),
            T.tensor(self.values, dtype=T.float32),
            T.tensor(self.next_values, dtype=T.float32),
            T.tensor(self.rewards, dtype=T.float32),
            T.tensor(self.dones, dtype=T.int64),
            batch_indices,
        )
