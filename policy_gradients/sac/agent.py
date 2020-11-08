import gym  # type: ignore
import numpy as np  # type: ignore
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from core import Agent, Hyperparameters
from replay_buffer import Action, Observation, ReplayBuffer
from utils import update_target_network

from sac.actor import Actor
from sac.critic import Critic
from sac.value import Value


class SAC(Agent):
    def __init__(self, hyperparameters: Hyperparameters) -> None:
        super(SAC, self).__init__()
        self.gamma = hyperparameters.gamma
        self.reward_scale = hyperparameters.reward_scale
        self.tau = hyperparameters.tau
        self.epsilon = hyperparameters.epsilon

        in_features = hyperparameters.env.observation_space.shape[0]
        action_dims = hyperparameters.env.action_space.shape[0]

        self.batch_size = hyperparameters.batch_size
        self.replay_buffer = ReplayBuffer(
            hyperparameters.replay_buffer_capacity, (in_features,), action_dims
        )

        self.actor = Actor(in_features, action_dims, epsilon=self.epsilon).to(
            self.device
        )
        self.critic_1 = Critic(in_features, action_dims)
        self.critic_2 = Critic(in_features, action_dims)
        self.value = Value(in_features)
        self.value_target = Value(in_features)

        self.update_value_target(tau=1)

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
            Q1 = self.critic_1(observation, action)
            Q2 = self.critic_2(observation, action)
            Q = T.minimum(Q1, Q2)
            V_target = Q - log_probability

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
            Q_target = self.reward_scale * reward + self.gamma * V_

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