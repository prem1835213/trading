import random
import torch
import torch.nn as nn
from collections import namedtuple, deque

class ReplayMemory(object):

    def __init__(self, memory_length=10):
        self.memory = deque([], maxlen=memory_length)

    def add_info(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN_Agent(nn.Module):

    def __init__(self, memory_length, state_dim, num_actions):
        super(DQN_Agent, self).__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.memory_length = memory_length
        self.memory = ReplayMemory(memory_length=self.memory_length)

        self.model = nn.Sequential(
            nn.Linear(self.state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.num_actions)
        )

    def forward(self, state):
        return self.model(state)

    def remember(self, sequence):
        self.memory.add_info(sequence)
