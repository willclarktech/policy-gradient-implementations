from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import gym  # type: ignore
import numpy as np  # type: ignore
import pybullet_envs  # type: ignore # pylint: disable=unused-import
import torch as T

from policy_gradients.utils import plot_returns


class Hyperparameters:
    # pylint: disable=invalid-name,too-few-public-methods,too-many-arguments,too-many-instance-attributes,too-many-locals
    def __init__(
        self,
        seed: Optional[int] = None,
        algorithm: str = "",
        env_name: str = "",
        n_episodes: int = 0,
        log_period: int = 1,
        hidden_features: Optional[List[int]] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        tau: float = 1.0,
        d: int = 1,
        batch_size: int = 1,
        replay_buffer_capacity: int = 0,
        reward_scale: float = 1.0,
        epsilon: float = 0.0,
        noise: float = 0.0,
        noise_clip: float = 0.0,
        l2_weight_decay: float = 0.0,
        save_dir: Optional[str] = None,
    ) -> None:
        self.seed = seed
        self.algorithm = algorithm
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.env.seed(self.seed)

        self.n_episodes = n_episodes
        self.log_period = log_period

        self.hidden_features = hidden_features if hidden_features is not None else []
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.d = d

        self.batch_size = batch_size
        self.replay_buffer_capacity = replay_buffer_capacity
        self.reward_scale = reward_scale

        self.epsilon = epsilon
        self.noise = noise
        self.noise_clip = noise_clip

        self.l2_weight_decay = l2_weight_decay

        self.save_dir = save_dir


class BaseAgent(metaclass=ABCMeta):
    def __init__(self, hyperparameters: Hyperparameters) -> None:
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.algorithm = hyperparameters.algorithm
        self.env_name = hyperparameters.env_name

    def process(
        self,
        observation: Union[List[List[float]], np.ndarray, T.Tensor],
        dtype: T.dtype = T.float32,
    ) -> T.Tensor:
        # pylint: disable=not-callable
        tensor = (
            observation.clone().detach()
            if isinstance(observation, T.Tensor)
            else T.tensor(observation, dtype=dtype)
        )
        return tensor.to(self.device)
        # pylint: enable=not-callable

    def get_savefile_name(self, dirname: str, component: str) -> str:
        return f"{dirname}/{self.algorithm}_{self.env_name}/{component}.zip"

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def eval(self) -> None:
        pass

    @abstractmethod
    def load(self, load_dir: str) -> None:
        pass

    @abstractmethod
    def save(self, save_dir: str) -> None:
        pass


GenericAgent = TypeVar("GenericAgent", bound=BaseAgent)
EpisodeRunner = Callable[
    [GenericAgent, Hyperparameters, Optional[bool], Optional[bool]], float
]

# pylint: disable=too-many-arguments
def train(
    agent: GenericAgent,
    hyperparameters: Hyperparameters,
    run_episode: EpisodeRunner,
    save_dir: Optional[str] = None,
    should_render: Optional[bool] = False,
    should_eval: Optional[bool] = False,
) -> None:
    n_episodes = hyperparameters.n_episodes
    log_period = hyperparameters.log_period

    returns = []
    average_returns = []

    for i in range(1, n_episodes + 1):
        ret = run_episode(agent, hyperparameters, should_render, should_eval)

        returns.append(ret)
        average_return = np.mean(returns[-100:])
        average_returns.append(average_return)

        if i % log_period == 0:
            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] Episode {i}; Return {ret}; Average return {average_return}"
            )

    if save_dir is not None:
        agent.save(save_dir)

    plot_returns(returns, average_returns)


class Algorithm:
    # pylint: disable=invalid-name,too-few-public-methods
    def __init__(
        self,
        Agent: Type[GenericAgent],
        default_hyperparameters: Callable[[], Dict[str, Any]],
        run_episode: EpisodeRunner,
    ):
        self.Agent = Agent
        self.default_hyperparameters = default_hyperparameters
        self.run_episode = run_episode
