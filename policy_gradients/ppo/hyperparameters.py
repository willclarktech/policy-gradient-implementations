from typing import Any, Dict


def default_hyperparameters() -> Dict[str, Any]:
    return dict(
        algorithm="ppo",
        env_name="InvertedPendulumBulletEnv-v0",
        n_episodes=300,
        log_period=1,
        hidden_features=[256, 256],
        alpha=3e-4,
        epsilon=0.2,
        gamma=0.99,
        lam=0.95,
        T=10,
        K=4,
        batch_size=5,
        N=1,
        c1=0.5,
        c2=0.01,
        seed=None,
    )
