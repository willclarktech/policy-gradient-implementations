from typing import Any, Dict


def default_hyperparameters() -> Dict[str, Any]:
    return dict(
        algorithm="human", env_name="LunarLander-v2", n_episodes=1, log_period=1
    )
