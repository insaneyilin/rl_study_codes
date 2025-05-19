import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):

    def __init__(self, input_shape, hidden_dim, num_actions):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[1],
                               out_channels=32,
                               kernel_size=8,
                               stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=4,
                               stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        conv_out_size = self._get_conv_out(input_shape)

        self.fc1 = nn.Linear(conv_out_size, hidden_dim)

        # Dueling DQN
        self.value_stream = nn.Linear(hidden_dim, 1)
        self.advantage_stream = nn.Linear(hidden_dim, num_actions)

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape[1:])
        o = self.conv1(o)
        o = self.bn1(o)

        o = self.conv2(o)
        o = self.bn2(o)

        o = self.conv3(o)
        o = self.bn3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        # Shared feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        # State value, [batch_size, 1]
        value = self.value_stream(x)

        # Advantage values for each action, [batch_size, num_actions]
        advantage = self.advantage_stream(x)

        # Combine streams using dueling architecture formula:
        # Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,a)))
        # This helps with identifiability and stable learning
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))
