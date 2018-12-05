import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetworkDense(nn.Module):
    def __init__(self, o_dim, u_dim, dueling):
        super(QNetworkDense, self).__init__()

        self._dueling = dueling
        self.fc_1 = nn.Linear(o_dim, 256)
        self.fc_2 = nn.Linear(256, 256)
        self.a_fc = nn.Linear(256, u_dim)
        if dueling:
            self.v_fc = nn.Linear(256, 1)

    def forward(self, state):
        x = F.relu(self.fc_1(state))
        x = F.relu(self.fc_2(x))
        a_x = self.a_fc(x)
        if self._dueling:
            v_x = self.v_fc(x)
            return v_x + (a_x - torch.mean(a_x, dim=1).view(-1, 1))
        return a_x


class QNetworkConv(nn.Module):
    def __init__(self, o_dim, u_dim):
        super(QNetworkConv, self).__init__()

        self.fc_1 = nn.Linear(o_dim, 256)
        self.fc_2 = nn.Linear(256, 256)
        self.a_fc_1 = nn.Linear(256, 128)
        self.v_fc_1 = nn.Linear(256, 128)
        self.a_fc_2 = nn.Linear(128, u_dim)
        self.v_fc_2 = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.fc_1(state))
        x = F.relu(self.fc_2(x))
        a_x = F.relu(self.a_fc_1(x))
        v_x = F.relu(self.v_fc_1(x))
        a_x = self.a_fc_2(a_x)
        v_x = self.v_fc_2(v_x)

        return v_x + (a_x - torch.mean(a_x, dim=1).view(-1, 1))