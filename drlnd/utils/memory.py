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
        :param capacity: int, set size of the buffer
        """
        self.capacity = capacity
        self._batch_size = batch_size

        self._observation1_buffer = deque()
        self._action_buffer = deque()
        self._reward_buffer = deque()
        self._observation2_buffer = deque()
        self._terminal_buffer = deque()

    def add(self, state, action, reward, next_state, done):
        """
        Add transition to replay buffer
        :param state: np.ndarray, observation in step t
        :param action: np.ndarray, action in step t
        :param reward: float, reward in step t
        :param next_state: np.ndarray, observation in step t+1
        :param done: boolean, signal if current state is terminal
        """
        self._add_to_buffer(self._observation1_buffer, state)
        self._add_to_buffer(self._action_buffer, action)
        self._add_to_buffer(self._reward_buffer, reward)
        self._add_to_buffer(self._observation2_buffer, next_state)
        self._add_to_buffer(self._terminal_buffer, done)

    def sample(self):
        """
        Sample minibatch from transition stored in replay buffer.
        :return: dict, minibatch
        """
        idxs = np.random.randint((self.size - 1), size=self._batch_size)
        batch = dict()
        batch['obs1'] = self._prepare_batch(self._observation1_buffer, idxs)
        batch['u'] = self._prepare_batch(self._action_buffer, idxs)
        batch['r'] = self._prepare_batch(self._reward_buffer, idxs)
        batch['obs2'] = self._prepare_batch(self._observation2_buffer, idxs)
        batch['d'] = self._prepare_batch(self._terminal_buffer, idxs)

        return batch

    @property
    def size(self):
        return len(self._observation1_buffer)

    def _add_to_buffer(self, buffer, value):
        if self.size >= self.capacity:
            buffer.popleft()
        buffer.append(value)

    @staticmethod
    def _prepare_batch(input_buffer, idxs):
        input_buffer = np.vstack(np.take(input_buffer, idxs, axis=0))
        return torch.from_numpy(input_buffer)
