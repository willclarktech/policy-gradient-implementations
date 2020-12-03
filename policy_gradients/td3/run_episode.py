from typing import Optional

from policy_gradients.core import Hyperparameters
from policy_gradients.td3.agent import Agent


def run_episode(
    agent: Agent,
    hyperparameters: Hyperparameters,
    should_render: Optional[bool] = False,
    should_eval: Optional[bool] = False,
) -> float:
    env = hyperparameters.env
    # Necessary for pybullet envs
    if should_render:
        env.render()

    observation = env.reset()
    done = False
    ret = 0.0
    step = 0

    if should_render:
        env.render()

    while not done:
        step += 1
        action = agent.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        ret += reward

        if not should_eval:
            agent.remember(observation, action, reward, observation_, done)
            agent.learn(step)

        observation = observation_

        if should_render:
            env.render()

    return ret
