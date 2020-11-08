import gym  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

from utils import attempt_with_screen, plot_returns

from td3.agent import Agent

n_episodes = 1000
# n_episodes = 1500
log_period = 1
env = gym.make("LunarLanderContinuous-v2")
# env = gym.make("BipedalWalker-v2")
agent = Agent(env)

returns = []
average_returns = []

for i in range(1, n_episodes + 1):
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

        if i == n_episodes:
            attempt_with_screen(env.render)

    returns.append(ret)
    average_return = np.mean(returns[-100:])
    average_returns.append(average_return)

    if i % log_period == 0:
        print(f"Episode {i}; Return: {ret}; Average return {average_return}")

attempt_with_screen(lambda: plot_returns(returns, average_returns))
