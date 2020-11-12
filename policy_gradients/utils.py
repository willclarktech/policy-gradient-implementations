import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import os
import random
import torch as T
import torch.nn as nn
from typing import Any, Callable, List, Optional


def attempt_with_screen(f: Callable[[], Any]) -> None:
    try:
        f()
    except:
        print("Screen unavailable")


def plot_returns(
    returns: List[float],
    average_returns: Optional[List[float]] = None,
    period: int = 100,
) -> None:
    if average_returns is None:
        average_returns = [
            np.mean(returns[max(0, i - period) : i]) for i, _ in enumerate(returns)
        ]

    plt.figure()
    plt.plot(returns, marker="x", label="returns")
    plt.plot(average_returns, marker="o", label="average")
    plt.title("Average return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.grid()
    plt.show()


def set_seed(seed: int) -> None:
    T.set_deterministic(True)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    T.manual_seed(seed)
    T.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def update_target_network(
    target_network: nn.Module, online_network: nn.Module, tau: float
) -> None:
    for target_param, online_param in zip(
        target_network.parameters(), online_network.parameters()
    ):
        target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)
    target_network.eval()
