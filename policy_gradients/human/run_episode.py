from time import sleep

from policy_gradients.core import Hyperparameters, TrainOptions
from policy_gradients.human.agent import Agent


def run_episode(
    agent: Agent,
    hyperparameters: Hyperparameters,
    _options: TrainOptions,
) -> float:
    env = hyperparameters.env

    # Necessary for pybullet envs
    env.render()

    env.reset()
    done = False
    ret = 0.0

    env.render()

    env.unwrapped.viewer.window.on_key_press = agent.handle_key_press
    env.unwrapped.viewer.window.on_key_release = agent.handle_key_release

    while not done:
        if not agent.is_paused:
            action = agent.choose_action()
            _, reward, done, _ = env.step(action)
            ret += reward

        env.render()
        sleep(0.05)

    return ret
