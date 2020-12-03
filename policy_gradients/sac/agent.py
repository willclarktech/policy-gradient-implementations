from gym import spaces  # type: ignore
import numpy as np  # type: ignore
import torch as T
import torch.nn.functional as F

from policy_gradients.core import BaseAgent, Hyperparameters
from policy_gradients.replay_buffer import ReplayBuffer
from policy_gradients.utils import update_target_network

from policy_gradients.sac.actor import Actor
from policy_gradients.sac.critic import Critic
from policy_gradients.sac.value import Value


class Agent(BaseAgent):
    # pylint: disable=invalid-name,too-many-arguments,too-many-instance-attributes
    def __init__(self, hyperparameters: Hyperparameters) -> None:
        super().__init__(hyperparameters)
        self.gamma = hyperparameters.gamma
        self.reward_scale = hyperparameters.reward_scale
        self.tau = hyperparameters.tau
        self.epsilon = hyperparameters.epsilon
        alpha = hyperparameters.alpha

        env = hyperparameters.env
        if not isinstance(env.action_space, spaces.Box):
            raise ValueError("This agent only supports box action spaces")
        self.min_action = env.action_space.low
        self.max_action = env.action_space.high
        in_dims = env.observation_space.shape
        action_dims = env.action_space.shape
        hidden_features = hyperparameters.hidden_features

        self.batch_size = hyperparameters.batch_size
        self.replay_buffer = ReplayBuffer(
            hyperparameters.replay_buffer_capacity,
            in_dims,
            action_dims,
            hyperparameters.seed,
        )

        self.actor = Actor(
            in_dims[0],
            action_dims[0],
            hidden_features,
            alpha=alpha,
            epsilon=self.epsilon,
        ).to(self.device)
        self.critic_1 = Critic(
            in_dims[0], action_dims[0], hidden_features, alpha=alpha
        ).to(self.device)
        self.critic_2 = Critic(
            in_dims[0], action_dims[0], hidden_features, alpha=alpha
        ).to(self.device)
        self.value = Value(in_dims[0], hidden_features, alpha).to(self.device)
        self.value_target = Value(in_dims[0], hidden_features, alpha).to(self.device)

        self.update_value_target(tau=1)

    def choose_action(self, observation: np.ndarray) -> np.ndarray:
        inp = self.process([observation])
        raw_action, _ = self.actor.sample(inp, reparameterize=False)
        detached_action = raw_action.cpu().detach().numpy()[0]
        return detached_action.clip(self.min_action, self.max_action)

    def remember(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        observation_: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.store_transition(
            observation, action, reward, observation_, done
        )

    def update_value(self, observation: T.Tensor) -> None:
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.value.train()

        with T.no_grad():
            action, log_probability = self.actor.sample(
                observation, reparameterize=False
            )
            Q1 = self.critic_1(observation, action)
            Q2 = self.critic_2(observation, action)
            Q = T.minimum(Q1, Q2)
            V_target = Q - log_probability

        self.value.optimizer.zero_grad()
        V = self.value(observation)
        V_loss = 0.5 * F.mse_loss(V.flatten(), V_target.flatten())
        V_loss.backward(retain_graph=True)
        self.value.optimizer.step()

    def update_critics(
        self,
        observation: T.Tensor,
        action: T.Tensor,
        reward: T.Tensor,
        observation_: T.Tensor,
        done: T.Tensor,
    ) -> None:
        self.critic_1.train()
        self.critic_2.train()

        with T.no_grad():
            V_ = self.value_target(observation_) * (1 - done)
            Q_target = self.reward_scale * reward + self.gamma * V_

        for critic in [self.critic_1, self.critic_2]:
            critic.train()
            critic.optimizer.zero_grad()
            Q = critic(self.process(observation), self.process(action))
            critic_loss = 0.5 * F.mse_loss(Q.flatten(), Q_target.flatten())
            critic_loss.backward(retain_graph=True)
            critic.optimizer.step()

    def update_actor(self, observation: T.Tensor) -> None:
        self.actor.train()
        self.critic_1.eval()
        self.critic_2.eval()

        (
            action_current_reparameterized,
            log_probability_reparameterized,
        ) = self.actor.sample(observation, reparameterize=True)
        Q1 = self.critic_1(observation, action_current_reparameterized)
        Q2 = self.critic_2(observation, action_current_reparameterized)
        Q = T.minimum(Q1, Q2)

        self.actor.optimizer.zero_grad()
        actor_loss = (log_probability_reparameterized.flatten() - Q.flatten()).mean()
        actor_loss.backward()
        self.actor.optimizer.step()

    def update_value_target(self, tau: float) -> None:
        update_target_network(self.value_target, self.value, tau)

    def learn(self) -> None:
        if self.replay_buffer.size < self.batch_size:
            return

        observation, action, reward, observation_, done = [
            self.process(t) for t in self.replay_buffer.sample(self.batch_size)
        ]

        self.update_value(observation)
        self.update_critics(observation, action, reward, observation_, done)
        self.update_actor(observation)
        self.update_value_target(tau=self.tau)

    def train(self) -> None:
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.value.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.value.eval()

    def load(self, load_dir: str) -> None:
        actor_state_dict = T.load(
            self.get_savefile_name(load_dir, "actor"), map_location=self.device
        )
        critic_1_state_dict = T.load(
            self.get_savefile_name(load_dir, "critic_1"), map_location=self.device
        )
        critic_2_state_dict = T.load(
            self.get_savefile_name(load_dir, "critic_2"), map_location=self.device
        )
        value_state_dict = T.load(
            self.get_savefile_name(load_dir, "value"), map_location=self.device
        )
        self.actor.load_state_dict(actor_state_dict)
        self.critic_1.load_state_dict(critic_1_state_dict)
        self.critic_2.load_state_dict(critic_2_state_dict)
        self.value.load_state_dict(value_state_dict)
        self.value_target.load_state_dict(value_state_dict)

    def save(self, save_dir: str) -> None:
        T.save(self.actor.state_dict(), self.get_savefile_name(save_dir, "actor"))
        T.save(self.critic_1.state_dict(), self.get_savefile_name(save_dir, "critic_1"))
        T.save(self.critic_2.state_dict(), self.get_savefile_name(save_dir, "critic_2"))
        T.save(self.value.state_dict(), self.get_savefile_name(save_dir, "value"))
