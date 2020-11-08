import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

from core import Hyperparameters
from utils import plot_returns

from reinforce.agent import Reinforce


def train(agent: Reinforce, hyperparameters: Hyperparameters) -> None:
    env = hyperparameters.env
    n_episodes = hyperparameters.n_episodes
    log_period = hyperparameters.log_period

    returns = []
    average_returns = []

    for i in range(1, n_episodes + 1):
        agent.reset()
        observation = env.reset()
        done = False
        ret = 0.0

        while not done:
            action, log_probability = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            ret += reward
            agent.remember(log_probability, reward)
            observation = observation_

            if i == n_episodes:
                env.render()

        returns.append(ret)
        average_return = np.mean(returns[-100:])
        average_returns.append(average_return)
        agent.update()

        if i % log_period == 0:
            print(f"Episode {i}; Average return {average_return}")

    plot_returns(returns, average_returns)
