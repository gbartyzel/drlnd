import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from drlnd.utils.memory import ReplayMemory
from drlnd.p1_navigation.dqn.model import QNetworkDense


class Agent(object):
    def __init__(self, state_dim, action_dim, lrate, gamma, n_step, n_step_annealing, epsilon_min,
                 update_frequency, target_update_frequency, buffer_size, batch_size,
                 warm_up_steps, use_double_q, use_dueling, use_noisynet, logdir):
        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")
        self._action_dim = action_dim
        self._state_dim = state_dim
        self.gamma = gamma
        self.n_step = n_step

        self._update_freq = update_frequency
        self._tau = 1.0 / target_update_frequency
        self._batch_size = batch_size
        self._warm_up_steps = warm_up_steps

        self._use_double_q = use_double_q
        self._use_noisynet = use_noisynet

        self._epsilon = 1.0
        self._epsilon_min = epsilon_min
        self._epsilon_decay = (self._epsilon - self._epsilon_min) / n_step_annealing
        self.step = 0

        self.checkpoint_path = os.path.join(logdir, "checkpoint.pth")

        self._dqn = QNetworkDense(
            state_dim, action_dim, use_dueling, use_noisynet).to(self._device)
        self._dqn_target = QNetworkDense(
            state_dim, action_dim, use_dueling, use_noisynet).to(self._device)

        self.load_model()
        self._dqn_target.load_state_dict(self._dqn.state_dict())
        self._dqn_target.eval()

        self._memory = ReplayMemory(buffer_size, batch_size, state_dim, action_dim)

        self._optim = optim.Adam(self._dqn.parameters(), lr=lrate)

    def act(self, state, train=False):
        self._epsilon -= self._epsilon_decay
        self._epsilon = max(self._epsilon, self._epsilon_min)
        self.step += 1

        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._dqn.eval()
        with torch.no_grad():
            action_values = self._dqn(state)
        self._dqn.train()

        if np.random.rand() > self._epsilon or not train or self._use_noisynet:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.randint(self._action_dim)

    def observe(self, state, action, reward, next_state, done):
        self._memory.push(state, action, reward, next_state, done)
        if self.step % self._update_freq == 0:
            if self._memory.size >= self._warm_up_steps:
                self._learn()

    def _learn(self):
        train_batch = self._memory.sample(self._device)
        state_batch = train_batch['obs1']
        action_batch = train_batch['u'].long()
        reward_batch = train_batch['r']
        next_state_batch = train_batch['obs2']
        done_batch = train_batch['d']

        if self._use_double_q:
            next_actions = self._dqn(next_state_batch).detach().argmax(1).view(-1, 1)
            target_next_q = self._dqn_target(next_state_batch).detach().gather(1, next_actions)
        else:
            target_next_q = self._dqn_target(next_state_batch).detach().max(1)[0].view(-1, 1)

        expected_q = self._dqn(state_batch).gather(1, action_batch)
        target_q = reward_batch + (1.0 - done_batch) * self.gamma ** self.n_step * target_next_q

        loss = F.smooth_l1_loss(expected_q, target_q)
        self._optim.zero_grad()
        loss.backward()
        for param in self._dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optim.step()

        for target_param, param in zip(self._dqn_target.parameters(), self._dqn.parameters()):
            target_param.data.copy_(self._tau * param.data + (1.0 - self._tau) * target_param.data)

    def save_model(self):
        torch.save(self._dqn.state_dict(), self.checkpoint_path)

    def load_model(self):
        if not os.path.isdir(os.path.split(self.checkpoint_path)[0]):
            os.makedirs(os.path.split(self.checkpoint_path)[0])
        if os.path.exists(self.checkpoint_path):
            self._dqn.load_state_dict(torch.load(self.checkpoint_path))
            print("Model found and loaded!")

    @property
    def parameters(self):
        return self._dqn.named_parameters()

    @property
    def target_parameters(self):
        return self._dqn_target.named_parameters()
