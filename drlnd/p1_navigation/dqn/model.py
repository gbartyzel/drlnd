import torch
import torch.nn as nn
import torch.nn.functional as F

from drlnd.utils.noisy_linear import FactorizedNoisyLinear


class QNetworkDense(nn.Module):
    def __init__(self, o_dim, u_dim, dueling=False, noisynet=False):
        super(QNetworkDense, self).__init__()

        layer = nn.Linear
        if noisynet:
            layer = FactorizedNoisyLinear

        self._dueling = dueling
        self.fc_1 = nn.Linear(o_dim, 128)
        self.a_fc_1 = layer(128, 64)
        self.a_fc_2 = layer(64, u_dim)
        if dueling:
            self.v_fc_1 = layer(128, 64)
            self.v_fc_2 = layer(64, 1)

    def forward(self, state):
        x = F.relu(self.fc_1(state))
        a_x = F.relu(self.a_fc_1(x))
        a_x = self.a_fc_2(a_x)
        if self._dueling:
            v_x = F.relu(self.v_fc_1(x))
            v_x = self.v_fc_2(v_x)
            return v_x + (a_x - torch.mean(a_x, dim=1).view(-1, 1))
        return a_x


class QNetworkConv(nn.Module):
    def __init__(self, channels, u_dim, dueling=False, noisynet=False):
        super(QNetworkConv, self).__init__()

        layer = nn.Linear
        if noisynet:
            layer = FactorizedNoisyLinear

        self.dueling = dueling
        self.conv_1 = nn.Conv2d(channels, 8, kernel_size=4, stride=4)
        self.conv_2 = nn.Conv2d(8, 16, kernel_size=4, stride=4)
        self.conv_3 = nn.Conv2d(16, 32, kernel_size=4, stride=4)
        self.size = 32

        self.a_fc_1 = layer(self.size, 16)
        self.a_fc_2 = layer(16, u_dim)
        if dueling:
            self.v_fc_1 = layer(self.size, 16)
            self.v_fc_2 = layer(16, 1)

    def forward(self, state):
        x = F.relu(self.conv_1(state))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = x.view(-1, self.size)
        a_x = F.relu(self.a_fc_1(x))
        a_x = self.a_fc_2(a_x)
        if self.dueling:
            v_x = F.relu(self.v_fc_1(x))
            v_x = self.v_fc_2(v_x)
            return v_x + (a_x - torch.mean(a_x, dim=1).view(-1, 1))
        return a_x
