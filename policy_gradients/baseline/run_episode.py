from policy_gradients.core import Hyperparameters, TrainOptions
from policy_gradients.baseline.agent import Agent


def run_episode(
    agent: Agent,
    hyperparameters: Hyperparameters,
    options: TrainOptions,
) -> float:
    env = hyperparameters.env
    should_render = options.should_render

    # Necessary for pybullet envs
    if should_render:
        env.render()

    env.reset()
    done = False
    ret = 0.0

    if should_render:
        env.render()

    while not done:
        action = agent.choose_action()
        _, reward, done, _ = env.step(action)
        ret += reward

        if should_render:
            env.render()

    return ret
