import torch
import random
import numpy as np
import torch.nn as nn
from torch.distributions import Normal

class Actor(nn.Module):

    def __init__(self, num_dimensions):
        super(Actor, self).__init__()
        self.num_dimensions = num_dimensions

        self.policy = nn.Sequential(
            nn.Linear(num_dimensions, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # need to create Parameter so that stds can update
        self.log_std = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, state):
        mean = self.policy(state)
        std = torch.clamp(self.log_std.exp(), 1e-3, 1)
        return Normal(mean, std)

class Critic(nn.Module):

    def __init__(self, num_dimensions):
        super(Critic, self).__init__()
        self.num_dimensions = num_dimensions

        self.v_func = nn.Sequential(
            nn.Linear(num_dimensions, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.v_func(state)
