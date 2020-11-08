#!/usr/bin/env python3
import argparse


def main(experiment: str) -> None:
    if experiment == "actor_critic":
        import actor_critic.__main__
    elif experiment == "ddpg":
        import ddpg.__main__
    elif experiment == "reinforce":
        import reinforce.__main__
    elif experiment == "sac":
        import sac.__main__
    elif experiment == "td3":
        import td3.__main__


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    args = parser.parse_args()
    main(args.experiment)
