from typing import Any, Dict


def default_hyperparameters() -> Dict[str, Any]:
    return dict(
        algorithm="actor_critic",
        env_name="LunarLander-v2",
        n_episodes=2000,
        log_period=1,
        hidden_features=[2048, 1536],
        alpha=5e-6,
        gamma=0.99,
        seed=None,
    )
