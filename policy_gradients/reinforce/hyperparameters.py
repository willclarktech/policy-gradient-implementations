from typing import Any, Dict


def default_hyperparameters() -> Dict[str, Any]:
    return dict(
        algorithm="reinforce",
        env_name="LunarLander-v2",
        n_episodes=3000,
        log_period=1,
        hidden_features=[128, 128],
        alpha=5e-4,
        gamma=0.99,
    )
