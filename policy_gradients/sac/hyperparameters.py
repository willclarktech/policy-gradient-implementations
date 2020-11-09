from typing import Optional

from core import Hyperparameters


def default_hyperparameters(seed: Optional[int]) -> Hyperparameters:
    return Hyperparameters(
        seed=seed,
        env_name="InvertedPendulumBulletEnv-v0",
        n_episodes=250,
        log_period=1,
        hidden_features=[256, 256],
        alpha=3e-4,
        gamma=0.99,
        tau=5e-3,
        epsilon=1e-6,
        batch_size=256,
        replay_buffer_capacity=1_000_000,
        reward_scale=2.0,
    )
