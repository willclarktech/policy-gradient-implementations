import gym  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

from utils import plot_returns

from actor_critic.agent import Agent

n_episodes = 2000
log_period = 1
env = gym.make("LunarLander-v2")
agent = Agent(env.observation_space.shape[0], env.action_space.n)

returns = []
average_returns = []

for i in range(1, n_episodes + 1):
    observation = env.reset()
    done = False
    ret = 0.0

    while not done:
        action, log_probability = agent.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        ret += reward
        agent.update(observation, log_probability, reward, done, observation_)
        observation = observation_

        if i == n_episodes:
            env.render()

    returns.append(ret)
    average_return = np.mean(returns[-100:])
    average_returns.append(average_return)

    if i % log_period == 0:
        print(f"Episode {i}; Average return {average_return}")

plot_returns(returns, average_returns)
