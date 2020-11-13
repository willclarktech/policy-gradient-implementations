from core import HyperparametersArgs


def default_hyperparameters() -> HyperparametersArgs:
    return HyperparametersArgs(
        env_name="LunarLanderContinuous-v2",
        n_episodes=1000,
        log_period=1,
        hidden_features=[256, 256],
        alpha=1e-3,
        gamma=0.99,
        tau=5e-3,
        d=2,
        batch_size=100,
        replay_buffer_capacity=1_000_000,
        noise=0.2,
        noise_clip=0.5,
    )
