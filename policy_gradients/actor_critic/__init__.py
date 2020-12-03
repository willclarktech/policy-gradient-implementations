from policy_gradients.core import Algorithm

from policy_gradients.actor_critic.agent import Agent
from policy_gradients.actor_critic.hyperparameters import default_hyperparameters
from policy_gradients.actor_critic.run_episode import run_episode

algorithm = Algorithm(Agent, default_hyperparameters, run_episode)

__all__ = ["algorithm"]
