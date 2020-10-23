import gym  # type: ignore

from agent import Agent

num_episodes = 50_000
log_period = 10_000

agent = Agent()
env = gym.make("Blackjack-v0")

for i in range(num_episodes):
    if i % log_period == 0:
        print(f"Starting episode {i}")

    agent.reset()
    observation = env.reset()
    done = False

    while not done:
        action = agent.policy(observation)
        observation_, reward, done, _ = env.step(action)
        agent.remember(observation, reward)
        observation = observation_

    agent.update_V()

print(agent.V[(21, 3, True)])  # almost certain to win
print(agent.V[(4, 1, False)])  # almost certain to lose
