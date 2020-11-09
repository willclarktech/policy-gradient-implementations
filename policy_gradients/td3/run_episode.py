import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

from core import Hyperparameters
from utils import attempt_with_screen, plot_returns

from td3.agent import Agent


def run_episode(agent: Agent, hyperparameters: Hyperparameters) -> float:
    env = hyperparameters.env
    observation = env.reset()
    done = False
    ret = 0.0
    step = 0

    while not done:
        step += 1
        action = agent.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        ret += reward
        agent.remember(observation, action, reward, observation_, done)
        observation = observation_
        agent.learn(step)

    return ret
