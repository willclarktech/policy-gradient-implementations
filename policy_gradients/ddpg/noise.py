from typing import Optional

import numpy as np  # type: ignore


class OrnsteinUhlenbeckNoise:
    # pylint: disable=invalid-name,too-many-arguments
    def __init__(
        self,
        mu: np.ndarray,
        sigma: float = 0.15,
        theta: float = 0.2,
        dt: float = 0.01,
        x0: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> None:
        if x0 is not None and not x0.shape == mu.shape:
            raise ValueError("x0 shape must match mu")

        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0 if x0 is not None else np.zeros_like(self.mu)
        self.x = self.x0
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> np.ndarray:
        self.x = (
            self.x
            + self.theta * (self.mu - self.x) * self.dt
            + self.sigma * np.sqrt(self.dt) * self.rng.normal(size=self.mu.shape)
        )
        return self.x

    def reset(self) -> None:
        self.x = self.x0
