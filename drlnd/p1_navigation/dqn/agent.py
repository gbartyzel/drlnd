import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from drlnd.utils.memory import ReplayMemory
from drlnd.p1_navigation.dqn.model import QNetworkDense, QNetworkConv


class Agent(object):
    def __init__(self, state_dim, action_dim, lrate, tau, gamma,
                 n_step_annealing, epsilon_min, update_frequency,
                 buffer_size, batch_size, use_double_q,
                 use_dueling, use_noisynet, logdir):
        """
        Initialize DQN agent.
        
        Params
        ======
            lrate (float): learning rate,
            tau (float): parameter for soft update,
            gamma (float): discount factor for reward,
            n_step_annealing (int): define max steps for exploration
            eps_min (float): minimum value of the epsilon,
            update_freq (int): update frequency for the agent,
            buffer_size (int): replay memory capacity,
            batch_size (int): size of the minibatch,
            double_q (bool): flag that enable double dqn,
            dueling (bool): flag that enable dueling dqn,
            model_path (string): path for saved model.
        """
        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._tau = tau
        self._gamma = gamma
        self._update_freq = update_frequency
        self._batch_size = batch_size
        self._use_double_q = use_double_q
        self._use_noisynet = use_noisynet

        self._epsilon = 1.0
        self._epsilon_min = epsilon_min
        self._epsilon_decay = (self._epsilon - self._epsilon_min) / n_step_annealing
        self.steps = 0

        self._checkpoint_path = os.path.join(logdir, "checkpoint.pth")

        self._dqn_main = QNetworkDense(state_dim, action_dim, use_dueling).to(self._device)
        self._dqn_target = QNetworkDense(state_dim, action_dim, use_dueling).to(self._device)
        self.load_model()
        self._dqn_target.load_state_dict(self._dqn_target.state_dict())
        self._memory = ReplayMemory(buffer_size, batch_size)

        self._optim = optim.Adam(self._dqn_main.parameters(), lrate)

    def act(self, state, train=False):
        """
        Returns actions for given state from environment
        
        Params
        ======
            state (array_like): current state
            train (bool): train flag for exploration
        """
        self._epsilon -= self._epsilon_decay
        self._epsilon = max(self._epsilon, self._epsilon_min)
        self.steps += 1

        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._dqn_main.eval()
        with torch.no_grad():
            action_values = self._dqn_main(state)
        self._dqn_main.train()

        if np.random.rand() > self._epsilon or not train:
            return np.argmax(action_values.cpu().data.numpy()), action_values.cpu().data.numpy()
        else:
            return np.random.randint(self._action_dim), action_values.cpu().data.numpy()

    def observe(self, state, action, reward, next_state, done):
        """
        Perform learning procedure.
        
        Params:
            state (array_like): current state,
            action (int): performed action,
            reward (float): reward received from environemnt,
            next_state (array_like): next state,
            done (bool): environment terminal flag.
        """
        self._memory.add(state, action, reward, next_state, done)
        if self.steps % self._update_freq == 0:
            if self._memory.size >= self._batch_size:
                self._learn()

    def _learn(self):
        train_batch = self._memory.sample()
        state_batch = train_batch['obs1'].to(self._device)
        action_batch = train_batch['u'].to(self._device)
        reward_batch = train_batch['r'].to(self._device)
        next_state_batch = train_batch['obs2'].to(self._device)
        done_batch = train_batch['d'].float().to(self._device)

        if self._use_double_q:
            next_actions = self._dqn_main(next_state_batch).detach().argmax(1).view(-1, 1)
            q_target_next = self._dqn_target(next_state_batch).detach().gather(1, next_actions)
        else:
            q_target_next = self._dqn_target(
                next_state_batch).max(1)[0].view(-1, 1)

        q_values = self._dqn_main(state_batch).gather(1, action_batch)
        q_target = reward_batch + (1.0 - done_batch) * self._gamma * q_target_next

        loss = F.smooth_l1_loss(q_values.squeeze(), q_target.squeeze())
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        self._update_target()

    def _update_target(self):
        for t_param, param in zip(self._dqn_target.parameters(), self._dqn_main.parameters()):
            t_param.data.copy_((1.0 - self._tau) * t_param.data + self._tau * param.data)

    def save_model(self):
        """
        Save model in path.
        """
        torch.save(self._dqn_main.state_dict(), self._checkpoint_path)

    def load_model(self):
        """
        Load model if exists.
        """
        if not os.path.isdir(os.path.split(self._checkpoint_path)[0]):
            os.makedirs(os.path.split(self._checkpoint_path)[0])
        if os.path.exists(self._checkpoint_path):
            self._dqn_main.load_state_dict(torch.load(self._checkpoint_path))
            print("Model found and loaded!")

    @property
    def parameters(self):
        return self._dqn_main.named_parameters()

    @property
    def target_parameters(self):
        return self._dqn_target.named_parameters()
