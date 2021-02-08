from abc import ABCMeta, abstractmethod
from datetime import datetime
import json
import os
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import gym  # type: ignore
import numpy as np  # type: ignore
import pybullet_envs  # type: ignore # pylint: disable=unused-import
import torch

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
        lam: float = 1.0,
        tau: float = 1.0,
        d: int = 1,
        K: int = 1,
        N: int = 1,
        T: int = 1,
        batch_size: int = 1,
        replay_buffer_capacity: int = 0,
        reward_scale: float = 1.0,
        epsilon: float = 0.0,
        noise: float = 0.0,
        noise_clip: float = 0.0,
        l2_weight_decay: float = 0.0,
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
        self.lam = lam
        self.tau = tau
        self.d = d
        self.K = K
        self.N = N
        self.T = T

        self.batch_size = batch_size
        self.replay_buffer_capacity = replay_buffer_capacity
        self.reward_scale = reward_scale

        self.epsilon = epsilon
        self.noise = noise
        self.noise_clip = noise_clip

        self.l2_weight_decay = l2_weight_decay

    def to_json(self) -> str:
        json_vars = dict(vars(self))
        json_vars.pop("env")
        return json.dumps(json_vars, indent=4, sort_keys=True)


class BaseAgent(metaclass=ABCMeta):
    def __init__(self, hyperparameters: Hyperparameters) -> None:
        self.hyperparameters = hyperparameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.algorithm = hyperparameters.algorithm
        self.env_name = hyperparameters.env_name

    def process(
        self,
        observation: Union[List[float], List[List[float]], np.ndarray, torch.Tensor],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        # pylint: disable=not-callable
        tensor = (
            observation.clone().detach()
            if isinstance(observation, torch.Tensor)
            else torch.tensor(observation, dtype=dtype)
        )
        return tensor.to(self.device)
        # pylint: enable=not-callable

    def get_savedir_name(self, dirname: str) -> str:
        return f"{dirname}/{self.algorithm}_{self.env_name}"

    def get_savefile_name(self, dirname: str, component: str) -> str:
        return f"{self.get_savedir_name(dirname)}/{component}.zip"

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def eval(self) -> None:
        pass

    @abstractmethod
    def load(self, load_dir: str) -> None:
        pass

    def save(self, save_dir: str) -> None:
        model_save_dir = self.get_savedir_name(save_dir)
        os.makedirs(model_save_dir, exist_ok=True)
        with open(
            f"{model_save_dir}/hyperparameters.json", "w"
        ) as hyperparameters_json:
            hyperparameters_json.write(self.hyperparameters.to_json())


GenericAgent = TypeVar("GenericAgent", bound=BaseAgent)
EpisodeRunner = Callable[[GenericAgent, Hyperparameters, bool, bool], float]


def train(
    agent: GenericAgent,
    hyperparameters: Hyperparameters,
    run_episode: EpisodeRunner,
    should_render: bool = False,
    should_eval: bool = False,
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
