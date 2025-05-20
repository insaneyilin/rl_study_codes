import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class BaseNet(nn.Module):

    def __init__(self, input_shape, hidden_dim):
        super(BaseNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[1],
                      out_channels=32,
                      kernel_size=8,
                      stride=4), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
                      stride=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                      stride=1), nn.BatchNorm2d(64), nn.ReLU())

        self.conv_out_size = self._get_conv_out(input_shape)
        self.fc1 = nn.Linear(self.conv_out_size, hidden_dim)

    def _get_conv_out(self, shape):
        with torch.no_grad():
            o = self.conv_layers(torch.zeros(1, *shape[1:]))
            return int(np.prod(o.size()))

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))


class ActorNet(BaseNet):

    def __init__(self, input_shape, hidden_dim, num_actions):
        super(ActorNet, self).__init__(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


class CriticNet(BaseNet):

    def __init__(self, input_shape, hidden_dim):
        super(CriticNet, self).__init__(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        values = self.fc2(x)
        return values


if __name__ == '__main__':
    input_shape = (1, 4, 175, 500)
    summary(ActorNet(input_shape, 128, 3))
    summary(CriticNet(input_shape, 128))
