import gym  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from agent import Agent

num_episodes = 50_000
log_period = 1000

agent = Agent()
env = gym.make("Blackjack-v0")
lose_draw_win_count = {-1: 0, 0: 0, 1: 0}
cumulative_lose_rates = []
cumulative_draw_rates = []
cumulative_win_rates = []

for i in range(1, num_episodes + 1):
    agent.reset()
    observation = env.reset()
    done = False

    while not done:
        action = agent.policy(observation)
        observation_, reward, done, _ = env.step(action)
        agent.remember(observation, action, reward)
        observation = observation_

    agent.update_Q()

    lose_draw_win_count[reward] += 1
    if i % log_period == 0:
        cumulative_lose_rates.append(lose_draw_win_count[-1] / i)
        cumulative_draw_rates.append(lose_draw_win_count[0] / i)
        cumulative_win_rates.append(lose_draw_win_count[1] / i)
        print(f"Episode {i} cumulative win ratio: {lose_draw_win_count[1]/i}")

print(agent.Q[((21, 3, True), 0)])  # almost certain to win
print(agent.Q[((21, 3, True), 1)])  # certain to lose
print(agent.Q[((4, 1, False), 0)])  # almost certain to lose

plt.figure()
plt.plot(cumulative_lose_rates, marker="o", label="lose")
plt.plot(cumulative_draw_rates, marker="x", label="draw")
plt.plot(cumulative_win_rates, marker=".", label="win")
plt.title("Cumulative win/draw/lose rates")
plt.xlabel("Episode")
plt.ylabel("Cumulative rate")
plt.legend()
plt.grid()
plt.show()
