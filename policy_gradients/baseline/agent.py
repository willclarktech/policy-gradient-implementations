from policy_gradients.core import BaseAgent, Hyperparameters


class Agent(BaseAgent):
    # pylint: disable=invalid-name,not-callable
    def __init__(self, hyperparameters: Hyperparameters,) -> None:
        super().__init__(hyperparameters)
        self.action_space = hyperparameters.env.action_space

    def choose_action(self) -> int:
        return self.action_space.sample()

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass

    def load(self, load_dir: str) -> None:
        pass

    def save(self, save_dir: str) -> None:
        pass
