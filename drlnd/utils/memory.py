import torch
import numpy as np


class RingBuffer(object):
    def __init__(self, capacity, dim):
        self._size = 0
        self._capacity = capacity
        self._buffer = np.zeros((capacity, dim), dtype=np.float32)

    def push(self, input):
        if self._size < self._capacity:
            self._size += 1
            self._buffer[self._size - 1, :] = input
        elif self._size == self._capacity:
            self._buffer = np.roll(self._buffer, shift=-1, axis=0)
            self._buffer[self._capacity - 1, :] = input

    def get_batch(self, batch_indexs):
        batch = np.take(self._buffer, batch_indexs, axis=0)
        if not batch.shape[1] > 1:
            return batch.squeeze(axis=1)
        return batch


class ReplayMemory(object):
    """
    Replay memory for reinforcement learning algorithms. Store agent's
    experience in fixed size buffer and sample transition of batch size for
    learning purpose.
    """

    def __init__(self, capacity, batch_size, state_dim, action_dim):
        """
        Initialize replay memory
        
        Params
        ======
            capacity (int): set size of the buffer,
            batch_size (int): set size of the minibatch
        """
        self._capacity = capacity
        self.batch_size = batch_size

        self._observation1_buffer = RingBuffer(capacity, state_dim)
        self._action_buffer = RingBuffer(capacity, action_dim)
        self._reward_buffer = RingBuffer(capacity, 1)
        self._observation2_buffer = RingBuffer(capacity, state_dim)
        self._terminal_buffer = RingBuffer(capacity, 1)

    def push(self, state, action, reward, next_state, done):
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
        self._observation1_buffer.push(state)
        self._action_buffer.push(action)
        self._reward_buffer.push(reward)
        self._observation2_buffer.push(next_state)
        self._terminal_buffer.push(done)

    def sample(self, device, idxs=None):
        """
        Return minibatch from transition stored in replay buffer.
        """
        if not idxs:
            idxs = np.random.randint((self.size - 1), size=self.batch_size)
        batch = dict()
        batch['obs1'] = torch.from_numpy(self._observation1_buffer.get_batch(idxs)).to(device)
        batch['u'] = torch.from_numpy(self._action_buffer.get_batch(idxs)).to(device)
        batch['r'] = torch.from_numpy(self._reward_buffer.get_batch(idxs)).to(device)
        batch['obs2'] = torch.from_numpy(self._observation2_buffer.get_batch(idxs)).to(device)
        batch['d'] = torch.from_numpy(self._terminal_buffer.get_batch(idxs)).to(device)

        return batch

    @property
    def size(self):
        return self._observation1_buffer._size
