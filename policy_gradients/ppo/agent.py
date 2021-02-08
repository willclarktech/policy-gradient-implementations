from typing import Tuple

from gym import spaces  # type: ignore
import numpy as np  # type: ignore
import torch as T

from policy_gradients.core import BaseAgent, Hyperparameters

from policy_gradients.ppo.actor import Actor
from policy_gradients.ppo.critic import Critic
from policy_gradients.ppo.memory import PPOMemory


class Agent(BaseAgent):
    # pylint: disable=invalid-name,too-many-arguments,too-many-instance-attributes
    def __init__(self, hyperparameters: Hyperparameters) -> None:
        super().__init__(hyperparameters)
        alpha = hyperparameters.alpha
        self.gamma = hyperparameters.gamma
        self.epsilon = hyperparameters.epsilon
        self.K = hyperparameters.K
        self.lam = hyperparameters.lam
        self.batch_size = hyperparameters.batch_size

        env = hyperparameters.env
        if not isinstance(env.action_space, spaces.Box):
            raise ValueError("This agent only supports box action spaces")
        in_dims = env.observation_space.shape
        action_dims = env.action_space.shape
        hidden_features = hyperparameters.hidden_features

        self.reset_memory()
        self.actor = Actor(in_dims[0], action_dims[0], hidden_features, alpha).to(
            self.device
        )
        self.critic = Critic(in_dims[0], hidden_features, alpha).to(self.device)

    def reset_memory(self) -> None:
        self.memory = PPOMemory(self.batch_size)

    def remember(
        self,
        observation: np.ndarray,
        action: T.Tensor,
        log_probability: T.Tensor,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.memory.store(observation, action, log_probability, value, reward, done)

    def choose_action(
        self, observation: np.ndarray
    ) -> Tuple[T.Tensor, T.Tensor, float]:
        inp = self.process([observation])
        action, log_probability = self.actor.sample(inp)
        value = self.critic(inp)
        return action, T.squeeze(log_probability), T.squeeze(value).item()

    def calculate_actor_loss(
        self,
        observations: T.Tensor,
        actions: T.Tensor,
        old_log_probabilities: T.Tensor,
        advantages: T.Tensor,
    ) -> T.Tensor:
        distribution = self.actor(observations)
        new_log_probabilities = distribution.log_prob(actions)
        ratio = new_log_probabilities.exp() / old_log_probabilities.exp()
        weighted_ratio = advantages * ratio
        clipped_weighted_ratio = advantages * T.clamp(
            ratio, 1.0 - self.epsilon, 1.0 + self.epsilon
        )
        return -T.min(weighted_ratio, clipped_weighted_ratio).mean()

    def calculate_critic_loss(
        self, observations: T.Tensor, values: T.Tensor, advantages: T.Tensor
    ) -> T.Tensor:
        returns = advantages + values
        critic_value = self.critic(observations).squeeze()
        return (returns - critic_value).pow(2).mean()

    def process_batch(
        self,
        observations: T.Tensor,
        actions: T.Tensor,
        old_log_probabilities: T.Tensor,
        advantages: T.Tensor,
        values: T.Tensor,
    ) -> None:
        actor_loss = self.calculate_actor_loss(
            observations, actions, old_log_probabilities, advantages
        )
        critic_loss = self.calculate_critic_loss(observations, values, advantages)
        total_loss = actor_loss + 0.5 * critic_loss
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()

    def learn(self) -> None:
        for _ in range(self.K):
            (
                observations,
                actions,
                old_log_probabilities,
                values,
                rewards,
                dones,
                batch_indices,
            ) = self.memory.sample()

            advantages = T.zeros(len(rewards), dtype=T.float32)
            for t in range(len(rewards) - 1):
                discount = 1.0
                advantage = T.scalar_tensor(0.0, dtype=T.float32)
                for i in range(t, len(rewards) - 1):
                    advantage += discount * (
                        rewards[i]
                        + self.gamma * values[i + 1] * (1 - int(dones[i].item()))
                        - values[i]
                    )
                    discount *= self.gamma * self.lam
                advantages[t] = advantage

            for batch in batch_indices:
                self.process_batch(
                    observations[batch],
                    actions[batch],
                    old_log_probabilities[batch],
                    advantages[batch],
                    values[batch],
                )

        self.reset_memory()

    def train(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic.eval()

    def load(self, load_dir: str) -> None:
        actor_state_dict = T.load(
            self.get_savefile_name(load_dir, "actor"), map_location=self.device
        )
        critic_state_dict = T.load(
            self.get_savefile_name(load_dir, "critic"), map_location=self.device
        )
        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)

    def save(self, save_dir: str) -> None:
        super().save(save_dir)
        T.save(self.actor.state_dict(), self.get_savefile_name(save_dir, "actor"))
        T.save(self.critic.state_dict(), self.get_savefile_name(save_dir, "critic"))
