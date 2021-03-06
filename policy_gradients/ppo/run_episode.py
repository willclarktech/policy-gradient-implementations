from typing import List

import gym  # type: ignore
import numpy as np  # type: ignore

from policy_gradients.core import Hyperparameters, TrainOptions
from policy_gradients.ppo.agent import Agent


def create_local_agent(hyperparameters: Hyperparameters, tmp_dir: str) -> Agent:
    agent = Agent(hyperparameters)
    agent.load(tmp_dir)
    return agent


# pylint: disable=invalid-name,too-many-locals
def run_episode(
    agent: Agent,
    hyperparameters: Hyperparameters,
    options: TrainOptions,
) -> float:
    N = hyperparameters.N
    T = hyperparameters.T
    env_name = hyperparameters.env_name
    should_eval = options.should_eval
    should_render = options.should_render
    tmp_dir = options.tmp_dir

    agent.save(tmp_dir)
    # Fast type-safe initialisation
    local_agents: List[Agent] = [agent] * 6

    returns: List[float] = []
    t = 0

    # NOTE: This could be parallelised
    for n in range(N):
        local_agents[n] = create_local_agent(hyperparameters, tmp_dir)
        env = gym.make(env_name)
        # Necessary for pybullet envs
        if should_render:
            env.render()

        observation = env.reset()
        done = False
        ret = 0.0

        if should_render:
            env.render()

        while not done:
            action, log_probability, value = local_agents[n].choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            ret += reward
            t += 1

            if not should_eval:
                value_ = agent.evaluate(observation_)
                agent.remember(
                    observation, action, log_probability, value, reward, done, value_
                )

                if t % T == 0:
                    agent.learn()
                    agent.save(tmp_dir)
                    local_agents[n] = create_local_agent(hyperparameters, tmp_dir)

            observation = observation_

            if should_render:
                env.render()

        returns.append(ret)

    # HACK: The runner would need to be refactored if this was parallelised
    return np.mean(returns)
