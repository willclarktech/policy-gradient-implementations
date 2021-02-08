from policy_gradients.core import Hyperparameters
from policy_gradients.ppo.agent import Agent

# pylint: disable=invalid-name,too-many-locals
def run_episode(
    agent: Agent,
    hyperparameters: Hyperparameters,
    should_render: bool = False,
    should_eval: bool = False,
) -> float:
    T = hyperparameters.T
    env = hyperparameters.env
    # Necessary for pybullet envs
    if should_render:
        env.render()

    observation = env.reset()
    done = False
    ret = 0.0
    t = 0

    if should_render:
        env.render()

    while not done:
        action, log_probability, value = agent.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        ret += reward
        t += 1

        if not should_eval:
            agent.remember(observation, action, log_probability, value, reward, done)

            if t % T == 0:
                agent.learn()

        observation = observation_

        if should_render:
            env.render()

    return ret
