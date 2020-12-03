from policy_gradients.core import Algorithm

from policy_gradients.ddpg.agent import Agent
from policy_gradients.ddpg.hyperparameters import default_hyperparameters
from policy_gradients.ddpg.run_episode import run_episode

algorithm = Algorithm(Agent, default_hyperparameters, run_episode)

__all__ = ["algorithm"]
