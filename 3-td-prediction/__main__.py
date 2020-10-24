import gym  # type: ignore

from agent import Agent

num_episodes = 5000
log_period = 1000

agent = Agent()
env = gym.make("CartPole-v1")

for i in range(1, num_episodes + 1):
    observation = env.reset()
    done = False

    while not done:
        action = agent.policy(observation)
        observation_, reward, done, _ = env.step(action)
        agent.update_V(observation, reward, observation_)
        observation = observation_

    if i % log_period == 0:
        print(f"Episode {i}: {agent.V}")
