from policy_gradients.core import Algorithm

from policy_gradients.baseline.agent import Agent
from policy_gradients.baseline.hyperparameters import default_hyperparameters
from policy_gradients.baseline.run_episode import run_episode

algorithm = Algorithm(Agent, default_hyperparameters, run_episode)

__all__ = ["algorithm"]
