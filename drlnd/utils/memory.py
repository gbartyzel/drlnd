import torch
import numpy as np
from collections import deque


class ReplayMemory(object):
    """
    Replay memory for reinforcement learning algorithms. Store agent's
    experience in fixed size buffer and sample transition of batch size for
    learning purpose.
    """
    def __init__(self, capacity, batch_size):
        """
        Initialize replay memory
        
        Params
        ======
            capacity (int): set size of the buffer,
            batch_size (int): set size of the minibatch
        """
        self.capacity = capacity
        self._batch_size = batch_size

        self._observation1_buffer = deque(maxlen=capacity)
        self._action_buffer = deque(maxlen=capacity)
        self._reward_buffer = deque(maxlen=capacity)
        self._observation2_buffer = deque(maxlen=capacity)
        self._terminal_buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Add transition to replay buffer
        
        Params
        ======
            state (array_like): observation in step t,
            action (array_like): action in step t,
            reward (float): reward in step t,
            next_state (array_like): observation in step t+1,
            done (bool): terminal signal from environment,
        """
        self._add_to_buffer(self._observation1_buffer, state)
        self._add_to_buffer(self._action_buffer, action)
        self._add_to_buffer(self._reward_buffer, reward)
        self._add_to_buffer(self._observation2_buffer, next_state)
        self._add_to_buffer(self._terminal_buffer, float(done))

    def sample(self, device):
        """
        Return minibatch from transition stored in replay buffer.
        """
        idxs = np.random.randint((self.size - 1), size=self._batch_size)
        batch = dict()
        batch['obs1'] = self._prepare_batch(self._observation1_buffer, idxs).float().to(device)
        batch['u'] = self._prepare_batch(self._action_buffer, idxs).to(device)
        batch['r'] = self._prepare_batch(self._reward_buffer, idxs).float().to(device)
        batch['obs2'] = self._prepare_batch(self._observation2_buffer, idxs).float().to(device)
        batch['d'] = self._prepare_batch(self._terminal_buffer, idxs).float().to(device)

        return batch

    @property
    def size(self):
        return len(self._observation1_buffer)

    def _add_to_buffer(self, buffer, value):
        if self.size >= self.capacity:
            buffer.popleft()
        if len(np.shape(value)) > 1:
            value = np.expand_dims(value, axis=0)
        buffer.append(value)

    @staticmethod
    def _prepare_batch(input_buffer, idxs):
        input_buffer = np.vstack(np.take(input_buffer, idxs, axis=0))
        return torch.from_numpy(input_buffer)
