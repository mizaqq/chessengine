import torch
import numpy as np
import pyspiel
import chess
from open_spiel.python import rl_environment

from src.utils.utils import piece_difference_from_tensor
from src.envs.open_spiel_env import OpenSpielEnv
import random


class WrappedEnv(OpenSpielEnv):
    pass


class VectorBuffer:
    def __init__(self, size: int):
        self.size = size
        self.states = torch.zeros(size, 20, 8, 8)
        self.actions = torch.zeros(size, 4674)

        self.actions_taken = torch.zeros(size, 1)
        self.rewards = torch.zeros(size, 1)
        self.dones = torch.zeros(size, 1)
        self.index = 0

    def add_current(self, state, action, action_taken, reward, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.actions_taken[self.index] = action_taken
        self.rewards[self.index] = reward
        self.dones[self.index] = done

    def update_index(self):
        self.index = (self.index + 1) % self.size

    def sample_n_steps(self, n: int):
        indices = random.randint(0, self.index - n - 1)
        states = self.states[indices : indices + n]
        actions = self.actions[indices : indices + n]
        actions_taken = self.actions_taken[indices : indices + n]
        rewards = self.rewards[indices : indices + n]
        dones = self.dones[indices : indices + n]
        next_states = self.states[indices + 1 : indices + n + 1]
        next_actions = self.actions[indices + 1 : indices + n + 1]
        next_rewards = self.rewards[indices + 1 : indices + n + 1]
        next_dones = self.dones[indices + 1 : indices + n + 1]

        return (
            states,
            actions,
            actions_taken,
            next_rewards - rewards,
            dones,
            next_states,
            next_actions,
            next_dones,
        )


class EnvSpawner:
    def __init__(self, num_envs: int):
        self.envs = []
        self.buffers = []
        for _ in range(num_envs):
            env = WrappedEnv()
            self.envs.append(env)
            self.buffers.append(VectorBuffer(100000))
            self.previous_actions = None
            self.previous_states = None

    def reset_all(self):
        for env in self.envs:
            env.reset()
        self.update_previous_actions()
        self.update_previous_states()

    def perform_actions(self, actions: list[list[int]]):
        for env, action in zip(self.envs, actions):
            env.step(action)

    def get_current_states(self):
        return torch.tensor(
            np.array([env.state() for env in self.envs], dtype=np.float32)
        )

    def get_current_actions(self):
        return torch.tensor(
            np.array([env.get_legal_actions() for env in self.envs], dtype=np.float32)
        )

    def get_previous_actions(self):
        return torch.tensor(
            np.array([env.previous_actions for env in self.envs], dtype=np.float32)
        )

    def get_previous_states(self):
        return torch.tensor(
            np.array([env.previous_states for env in self.envs], dtype=np.float32)
        )

    def update_previous_actions(self):
        self.previous_actions = self.get_current_actions()

    def update_previous_states(self):
        self.previous_states = self.get_current_states()

    def get_done(self):
        return torch.tensor(np.array([env.is_done() for env in self.envs]))

    def append_current_to_buffer(self, actions_taken):
        current_states = self.get_current_states()
        rewards = self.get_rewards()
        done = self.get_done()
        for i, (state, action, action_taken, reward, done) in enumerate(
            zip(current_states, self.previous_actions, actions_taken, rewards, done)
        ):
            self.buffers[i].add_current(state, action, action_taken, reward, done)
            self.buffers[i].update_index()

    def get_rewards(self, color="white"):
        current_states = self.get_current_states()
        rewards = piece_difference_from_tensor(
            current_states
        ) - piece_difference_from_tensor(self.previous_states)
        return rewards if color == "white" else -rewards

    def sample_n_steps(self, n: int):
        buffer = self.buffers[0]
        (
            states,
            actions,
            actions_taken,
            rewards,
            dones,
            next_states,
            next_actions,
            next_dones,
        ) = buffer.sample_n_steps(n)
        for buffer in self.buffers[1:]:
            s, a, at, nr, d, ns, na, nd = buffer.sample_n_steps(n)
            states = torch.cat([states, s], dim=0)
            actions = torch.cat([actions, a], dim=0)
            actions_taken = torch.cat([actions_taken, at], dim=0)
            rewards = torch.cat([rewards, nr], dim=0)
            dones = torch.cat([dones, d], dim=0)
            next_states = torch.cat([next_states, ns], dim=0)
            next_actions = torch.cat([next_actions, na], dim=0)
            next_dones = torch.cat([next_dones, nd], dim=0)
        return {
            "states": states,
            "actions": actions,
            "actions_taken": actions_taken,
            "rewards": rewards,
            "dones": dones,
            "next_states": next_states,
            "next_actions": next_actions,
            "next_dones": next_dones,
        }

    def generate_actions(self, policy, model, eps=0.05):
        current_states = self.get_current_states()
        actions = policy(model, current_states, self.get_current_actions(), eps)
        return torch.argmax(actions, dim=1).unsqueeze(1)

    def one_interation(self, actions: torch.Tensor):
        # White moves — skip envs that are already done
        self.update_previous_actions()
        self.update_previous_states()
        for env, action in zip(self.envs, actions.tolist()):
            if not env.is_done():
                env.step(action)

        # Black moves — skip envs that are now done (e.g. white just checkmated)
        for env in self.envs:
            if not env.is_done():
                legal = env.get_legal_actions().nonzero().tolist()
                env.step(random.choice(legal))

        self.append_current_to_buffer(actions)

    def move(self, actions: torch.Tensor):
        for env, action in zip(self.envs, actions.tolist()):
            if not env.is_done():
                env.step(action)
