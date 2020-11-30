from policy_gradients.core import Hyperparameters
from policy_gradients.sac.agent import Agent


def run_episode(
    agent: Agent, hyperparameters: Hyperparameters, should_render: bool = False
) -> float:
    env = hyperparameters.env
    # Necessary for pybullet envs
    if should_render:
        env.render()

    observation = env.reset()
    done = False
    ret = 0.0

    if should_render:
        env.render()

    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        ret += reward
        agent.remember(observation, action, reward, observation_, done)
        observation = observation_
        agent.learn()

        if should_render:
            env.render()

    return ret
