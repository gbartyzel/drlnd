import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, action_dim, state_dim):
        super(Actor, self).__init__()

        self._fc_1 = nn.Linear(state_dim, 200)
        self._fc_2 = nn.Linear(200, 150)
        self._action = nn.Linear(150, action_dim)

    def forward(self, state):
        x = F.relu(self._fc_1(state))
        x = F.relu(self._fc_2(x))
        x = torch.tanh(self._action(x))

        return x


class Critic(nn.Module):
    def __init__(self, action_dim, state_dim, double_critic=True):
        super(Critic, self).__init__()

        self._double_critic = double_critic

        self._critic_1_fc_1 = nn.Linear(state_dim + action_dim, 200)
        self._critic_1_fc_2 = nn.Linear(200, 150)
        self._critic_1_q_value = nn.Linear(150, 1)

        if double_critic:
            self._critic_2_fc_1 = nn.Linear(state_dim + action_dim, 200)
            self._critic_2_fc_2 = nn.Linear(200, 150)
            self._critic_2_q_value = nn.Linear(150, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)

        critic_1_x = F.relu(self._critic_1_fc_1(x))
        critic_1_x = F.relu(self._critic_1_fc_2(critic_1_x))
        critic_1_x = self._critic_1_q_value(critic_1_x)

        if self._double_critic:
            critic_2_x = F.relu(self._critic_2_fc_1(x))
            critic_2_x = F.relu(self._critic_2_fc_2(critic_2_x))
            critic_2_x = self._critic_2_q_value(critic_2_x)

            return critic_1_x, critic_2_x

        return critic_1_x

    def evaluate_q1(self, state, action):
        x = torch.cat((state, action), dim=1)

        x = F.relu(self._critic_1_fc_1(x))
        x = F.relu(self._critic_1_fc_2(x))
        return self._critic_1_q_value(x)

    def evaluate_q2(self, state, action):
        x = torch.cat((state, action), dim=1)

        x = F.relu(self._critic_2_fc_1(x))
        x = F.relu(self._critic_2_fc_2(x))
        return self._critic_2_q_value(x)

