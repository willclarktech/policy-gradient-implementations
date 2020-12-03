from policy_gradients.core import Hyperparameters
from policy_gradients.reinforce.agent import Agent


def run_episode(
    agent: Agent,
    hyperparameters: Hyperparameters,
    should_render: bool = False,
    should_eval: bool = False,
) -> float:
    env = hyperparameters.env
    # Necessary for pybullet envs
    if should_render:
        env.render()

    observation = env.reset()
    agent.reset()
    done = False
    ret = 0.0

    if should_render:
        env.render()

    while not done:
        action, log_probability = agent.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        ret += reward

        if not should_eval:
            agent.remember(log_probability, reward)

        observation = observation_

        if should_render:
            env.render()

    if not should_eval:
        agent.update()

    return ret
