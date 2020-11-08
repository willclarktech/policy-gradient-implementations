import numpy as np  # type: ignore

from core import Hyperparameters
from utils import attempt_with_screen, plot_returns

from sac.agent import Agent


def train(agent: Agent, hyperparameters: Hyperparameters) -> None:
    env = hyperparameters.env
    n_episodes = hyperparameters.n_episodes
    log_period = hyperparameters.log_period

    returns = []
    average_returns = []

    for i in range(1, n_episodes + 1):
        observation = env.reset()
        done = False
        ret = 0.0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            ret += reward
            agent.remember(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()

            if i == n_episodes:
                attempt_with_screen(lambda: env.render(mode="human"))

        returns.append(ret)
        average_return = np.mean(returns[-100:])
        average_returns.append(average_return)

        if i % log_period == 0:
            print(f"Episode {i}; Return: {ret}; Average return {average_return}")

    attempt_with_screen(lambda: plot_returns(returns, average_returns))
