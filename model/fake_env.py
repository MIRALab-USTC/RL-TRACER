import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from common import utils
from common.buffer import UnionReplayBuffer


class BaseEnv(object, metaclass=abc.ABCMeta):

    def __init__(
            self,
            dataset: dict,
            buffer: UnionReplayBuffer,
            device: torch.device = None,
            config = None
    ):
        # Setting hyperparameters
        self.device = device
        # Setting modules
        self.buffer = buffer
        self.config = config
        # Initialize buffer and transforms
        self._init_buffer(dataset)

    def _init_buffer(self, dataset):
        states = np.array(dataset["observations"], dtype=np.float32)
        next_states = np.array(dataset["next_observations"], dtype=np.float32)
        actions = np.array(dataset["actions"], dtype=np.float32)
        rewards = np.array(dataset["rewards"]).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)
        timestamp, j = [], 1
        for i, terminal in enumerate(terminals):
            if terminal != 1:
                timestamp.append(j)
                j += 1
            else:
                j = 1
        timestamp = np.array(timestamp)

        self.buffer.load_dataset(states, actions, rewards,
                                 next_states, terminals, timestamp)

    def step(self, state, action, deterministic=False):
        pass

    def learning_dynamics(self, ):
        pass

    def rollout_data(self, agent, rollout_batch_size, rollout_length):
        pass


class FakeEnv(BaseEnv):

    def __init__(
            self,
            dataset: dict,
            buffer: UnionReplayBuffer,
            device: torch.device = None,
            config = None
    ):
        super().__init__(dataset, buffer, device, config)

    def sample_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.buffer.batch_size

        batch_data = self.sample(batch_size)
        for k, v in batch_data.items():
            batch_data[k] = v.to(self.device)
        return batch_data

    def sample(self, batch_size):
        return self.buffer.sample(batch_size)

    def sample_state(self, batch_size):
        return self.buffer.sample_state(batch_size)

    def sample_all_data(self,):
        return self.buffer.sample_all_data()

    def add_batch(self, s, a, r, s2, done):
        self.buffer.add_batch(s, a, r, s2, done)
