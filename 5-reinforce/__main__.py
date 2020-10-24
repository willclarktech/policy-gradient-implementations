import gym  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

from agent import Agent

n_episodes = 3000
log_period = 100
env = gym.make("LunarLander-v2")
agent = Agent(env.observation_space.shape[0], env.action_space.n)

returns = []
average_returns = []

for i in range(1, n_episodes + 1):
    agent.reset()
    observation = env.reset()
    done = False
    ret = 0

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

plt.figure()
plt.plot(returns, marker="x", label="returns")
plt.plot(average_returns, marker="o", label="average")
plt.title("Average return")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.legend()
plt.grid()
plt.show()
