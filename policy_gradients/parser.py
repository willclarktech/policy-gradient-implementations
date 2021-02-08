import argparse
from typing import Iterable


def create_parser(algorithms: Iterable[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "policy_gradients",
        description="Run various implementations of policy gradient-based reinforcement learning algorithms.",
    )
    parser.add_argument("algorithm", help=f"choose from: {', '.join(algorithms)}")

    parser.add_argument("--seed", type=int, help="seed to use for reproducible results")
    parser.add_argument(
        "--env_name",
        "--env",
        help="choose an environment name from OpenAI’s gym or pybullet; defaults to a sensible choice",
    )
    parser.add_argument(
        "-n", "--n_episodes", type=int, help="number of training episodes"
    )
    parser.add_argument("--log_period", type=int, help="number of episodes per log")
    parser.add_argument(
        "--hidden_features",
        nargs="+",
        type=int,
        help="number of units in each hidden layer",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="primary learning rate, eg for actor if more than one learning rate is used",
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="secondary learning rate, eg for critic if more than one learning rate is used",
    )
    parser.add_argument("--gamma", type=float, help="discount factor")
    parser.add_argument("--tau", type=float, help="target network update weight")
    parser.add_argument(
        "-d", type=int, help="target network update frequency (episodes per update)"
    )
    parser.add_argument(
        "--batch_size", type=int, help="batch size to use during training"
    )
    parser.add_argument(
        "--replay_buffer_capacity", type=int, help="replay buffer capacity"
    )
    parser.add_argument("--reward_scale", type=float, help="reward scaling factor")
    parser.add_argument("--epsilon", type=float, help="exploration rate")
    parser.add_argument(
        "--noise",
        type=float,
        help="noise to use during action selection (eg for deterministic algorithms like TD3)",
    )
    parser.add_argument(
        "--noise_clip",
        type=float,
        help="noise clip to use during action selection (eg for deterministic algorithms like TD3)",
    )
    parser.add_argument("--save_dir", help="directory for saving model files")
    parser.add_argument(
        "--load_dir", help="directory for loading pre-trained model files"
    )
    parser.add_argument(
        "--render",
        default=False,
        action="store_true",
        help="whether to render the environment at each step",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action="store_true",
        help="whether to show stacktrace on error",
    )
    parser.add_argument(
        "--eval",
        default=False,
        action="store_true",
        help="whether to run the agent in evaluation mode (no training)",
    )

    return parser
