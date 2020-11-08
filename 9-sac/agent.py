import gym  # type: ignore
import numpy as np  # type: ignore
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from actor import Actor
from critic import Critic
from replay_buffer import Action, Observation, ReplayBuffer
from utils import update_target_network
from value import Value


class Agent:
    def __init__(
        self,
        env: gym.Env,
        batch_size: int = 256,
        replay_buffer_capacity: int = 1_000_000,
        gamma: float = 0.99,
        scale: float = 2.0,
        tau: float = 5e-3,
        epsilon: float = 1e-6,
    ) -> None:
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.gamma = 0.99
        self.scale = scale
        self.tau = tau
        self.epsilon = epsilon

        in_features = env.observation_space.shape[0]
        action_dims = env.action_space.shape[0]

        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(
            replay_buffer_capacity, (in_features,), action_dims
        )

        self.actor = Actor(in_features, action_dims, epsilon=epsilon).to(self.device)
        self.critic_1 = Critic(in_features, action_dims)
        self.critic_2 = Critic(in_features, action_dims)
        self.value = Value(in_features)
        self.value_target = Value(in_features)

        self.update_value_target(tau=1)

    def process(self, arr: np.ndarray, dtype=T.float32) -> T.Tensor:
        return T.tensor(arr, dtype=dtype).to(self.device)

    def choose_action(self, observation: Observation) -> np.ndarray:
        inp = self.process([observation])
        action, _ = self.actor.sample(inp, reparameterize=False)
        return action.cpu().detach().numpy()[0]

    def remember(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        observation_: Observation,
        done: bool,
    ) -> None:
        self.replay_buffer.store_transition(
            observation, action, reward, observation_, done
        )

    def update_value(self, observation) -> None:
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.value.train()

        with T.no_grad():
            action, log_probability = self.actor.sample(
                observation, reparameterize=False
            )
            Q1_current = self.critic_1(observation, action)
            Q2_current = self.critic_2(observation, action)
            Q_current = T.minimum(Q1_current, Q2_current)
            V_target = Q_current - log_probability

        self.value.optimizer.zero_grad()
        V = self.value(observation)
        V_loss = 0.5 * F.mse_loss(V.flatten(), V_target.flatten())
        V_loss.backward(retain_graph=True)
        self.value.optimizer.step()

    def update_critics(self, observation, action, reward, observation_, done) -> None:
        self.critic_1.train()
        self.critic_2.train()

        with T.no_grad():
            V_ = self.value_target(observation_) * (1 - done)
            Q_target = self.scale * reward + self.gamma * V_

        for critic in [self.critic_1, self.critic_2]:
            critic.train()
            critic.optimizer.zero_grad()
            Q = critic(observation, action)
            critic_loss = 0.5 * F.mse_loss(Q.flatten(), Q_target.flatten())
            critic_loss.backward(retain_graph=True)
            critic.optimizer.step()

    def update_actor(self, observation) -> None:
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
