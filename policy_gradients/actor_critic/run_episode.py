import numpy as np  # type: ignore

from policy_gradients.core import Hyperparameters
from policy_gradients.utils import plot_returns

from policy_gradients.actor_critic.agent import Agent


def run_episode(
    agent: Agent, hyperparameters: Hyperparameters, should_render: bool = False
) -> float:
    env = hyperparameters.env
    # Necessary for pybullet envs
    if should_render:
        env.render()

    observation = env.reset()
    done = False
    ret = 0.0

    if should_render:
        env.render()

    while not done:
        action, log_probability = agent.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        ret += reward
        agent.update(observation, log_probability, reward, done, observation_)
        observation = observation_

        if should_render:
            env.render()

    return ret
