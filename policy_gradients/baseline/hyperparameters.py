from typing import Any, Dict


def default_hyperparameters() -> Dict[str, Any]:
    return dict(env_name="CartPole-v1", n_episodes=100, log_period=1)
