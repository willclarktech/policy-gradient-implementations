from gym import spaces  # type: ignore

from policy_gradients.core import BaseAgent, Hyperparameters

KEYS = {"SPACE": 32}


def get_action_from_key(key: int) -> int:
    return int(key - ord("0"))


class Agent(BaseAgent):
    # pylint: disable=invalid-name,not-callable
    def __init__(
        self,
        hyperparameters: Hyperparameters,
    ) -> None:
        super().__init__(hyperparameters)

        env = hyperparameters.env
        if not isinstance(env.action_space, spaces.Discrete):
            raise ValueError("This agent only supports discrete action spaces")

        self.num_actions = env.action_space.n
        self.current_action = 0
        self.is_paused = False

        print("================================================================")
        print("Hello human agent!")
        print("Press and hold number keys to select action.")
        print("The default action is 0 when no key is pressed.")
        print(
            f"This environment supports actions between 0 and {self.num_actions - 1} inclusive."
        )
        print("Press SPACE to pause/unpause.")
        print("================================================================")

    def handle_key_press(self, key: int, _modifier: int) -> None:
        if key == KEYS["SPACE"]:
            self.is_paused = not self.is_paused
            return

        action = get_action_from_key(key)
        if not 0 <= action < self.num_actions:
            return

        self.current_action = action

    def handle_key_release(self, key: int, _modifier: int) -> None:
        action = get_action_from_key(key)
        if action == self.current_action:
            self.current_action = 0

    def choose_action(self) -> int:
        return self.current_action

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass

    def load(self, load_dir: str) -> None:
        pass

    def save(self, save_dir: str) -> None:
        pass
