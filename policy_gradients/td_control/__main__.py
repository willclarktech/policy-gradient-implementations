import gym  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

from utils import plot_returns

from td_control.agent import Agent
from td_control.digitizer import CartPoleDigitizer

num_episodes = 50_000
log_period = 1000

env = gym.make("CartPole-v1")
digitizer = CartPoleDigitizer()
agent = Agent(env.action_space.n, digitizer, epsilon_step=2.0 / num_episodes)
returns = []
average_returns = []

for i in range(1, num_episodes + 1):
    observation = env.reset()
    done = False
    ret = 0.0

    while not done:
        action = agent.policy(observation)
        observation_, reward, done, _ = env.step(action)
        ret += reward
        agent.update_Q(observation, action, reward, observation_)
        observation = observation_

    agent.decrement_epsilon()
    returns.append(ret)
    average_return = np.mean(returns[-100:])
    average_returns.append(average_return)

    if i % log_period == 0:
        print(
            f"Episode {i}; Average return: {average_return}; Epsilon: {agent.epsilon}"
        )

plot_returns(returns, average_returns)
