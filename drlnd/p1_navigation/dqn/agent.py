import numpy as np
import torch
import torch.optim as optim

from drlnd.utils.memory import ReplayMemory
from drlnd.p1_navigation.dqn.model import QNetworkDense


class Agent(object):
    def __init__(self, o_dim, u_dim, lrate, tau, gamma, eps_decay,
                 update_freq, buffer_size, batch_size, device):
        self._u_dim = u_dim
        self._o_dim = o_dim
        self._device = device
        self._tau = tau
        self._gamma = gamma
        self._update_freq = update_freq
        self._eps_decay = eps_decay
        self._batch_size = batch_size

        self._eps = 1.0
        self._steps = 0

        self._dqn_main = QNetworkDense(o_dim, u_dim).to(device)
        self._dqn_target = QNetworkDense(o_dim, u_dim).to(device)
        self._dqn_target.load_state_dict(self._dqn_target.state_dict())
        self._memory = ReplayMemory(buffer_size, batch_size)

        self._optim = optim.Adam(self._dqn_main.parameters(), lrate)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._dqn_main.eval()
        with torch.no_grad():
            action_values = self._dqn_main(state)
        self._dqn_main.train()

        if np.random.rand() > self._eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.randint(self._u_dim)

    def observe(self, state, action, reward, next_state, done):
        self._memory.add(state, action, reward, next_state, done)

        self._eps *= self._eps_decay
        self._steps += 1

        if self._steps % self._update_freq == 0:
            if self._memory.size >= self._batch_size:
                self._learn()

    def _learn(self):
        train_batch = self._memory.sample()
        state_batch = train_batch['obs1'].float().to(self._device)
        action_batch = train_batch['u'].float().to(self._device)
        reward_batch = train_batch['r'].float().to(self._device)
        next_state_batch = train_batch['obs2'].float().to(self._device)
        done_batch = train_batch['d'].float().to(self._device)

        with torch.no_grad():
            next_actions = torch.argmax(self._dqn_main(next_state_batch),
                                        dim=1).view(-1, 1)
            target_action_values = torch.gather(self._dqn_target(
                next_state_batch), 1, next_actions)

        action_values = torch.gather(
            self._dqn_main(state_batch), 1, action_batch)
        target = torch.squeeze(
            reward_batch + (1.0 - done_batch) * gamma * target_action_values)
        loss = F.mse_loss(torch.squeeze(action_values), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _update_target(self):
        for t_param, param in zip(self._dqn_target.parameters(),
                                  self._dqn_main.parameters()):
            t_param.data_copy_((1.0 - self._tau) * t_param + self._tau * param)