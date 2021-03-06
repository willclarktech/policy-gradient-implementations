from typing import Any, Dict


def default_hyperparameters() -> Dict[str, Any]:
    return dict(
        algorithm="ddpg",
        env_name="LunarLanderContinuous-v2",
        n_episodes=1000,
        log_period=1,
        hidden_features=[400, 300],
        alpha=1e-4,
        beta=1e-3,
        gamma=0.99,
        tau=1e-3,
        batch_size=64,
        replay_buffer_capacity=1_000_000,
        l2_weight_decay=0.01,
        seed=None,
    )
