import torch
import random


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
