import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Optional, Any
import hydra

from common.buffer import UnionReplayBuffer


class BaseEntroBuffer(object, metaclass=abc.ABCMeta):

    def __init__(
            self,
            clean_dataset: dict,
            attack_dataset: dict,
            clean_buffer: UnionReplayBuffer,
            attack_buffer: UnionReplayBuffer,
            device: Optional[torch.device] = None,
            config: Any = None
    ):
        # Setting hyperparameters
        self.device = device

        # Setting modules
        self.clean_buffer = clean_buffer
        self.attack_buffer = attack_buffer
        self.config = config

        # Initialize buffer and transforms
        self._init_clean_buffer(clean_dataset)
        self._init_attack_buffer(attack_dataset)

    def _init_clean_buffer(self, dataset):
        states = np.array(dataset["observations"], dtype=np.float32)
        next_states = np.array(dataset["next_observations"], dtype=np.float32)
        actions = np.array(dataset["actions"], dtype=np.float32)
        rewards = np.array(dataset["rewards"]).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)
        self.clean_buffer.load_dataset(states, actions, rewards, next_states, terminals)

    def _init_attack_buffer(self, dataset):
        states = np.array(dataset["observations"], dtype=np.float32)
        next_states = np.array(dataset["next_observations"], dtype=np.float32)
        actions = np.array(dataset["actions"], dtype=np.float32)
        rewards = np.array(dataset["rewards"]).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)
        self.attack_buffer.load_dataset(states, actions, rewards, next_states, terminals)


class EntroBuffer(BaseEntroBuffer):
    def __init__(
            self,
            clean_dataset: dict,
            attack_dataset: dict,
            clean_buffer: UnionReplayBuffer,
            attack_buffer: UnionReplayBuffer,
            device: Optional[torch.device] = None,
            config: Any = None
    ):
        super().__init__(clean_dataset, attack_dataset, clean_buffer, attack_buffer, device, config)

    def sample_data(self, real_ratio: float, batch_size: Optional[int] = None):
        ## real / (real + fake) = real_ratio
        if real_ratio > 0.0 and real_ratio < 1.0:
            real_bs = int(real_ratio * batch_size)
            fake_bs = batch_size - real_bs
            real_batch_data = self.sample('real', real_bs)
            fake_batch_data = self.sample('fake', fake_bs)
            batch_data = {}
            for key in real_batch_data.keys():
                batch_data[key] = torch.cat(
                    [real_batch_data[key], fake_batch_data[key]],
                dim=0).to(self.device)
            return batch_data

        elif real_ratio >= 1.0:
            batch_data = self.sample('real', batch_size)
        else:
            batch_data = self.sample('fake', batch_size)
        for k, v in batch_data.items():
            batch_data[k] = v.to(self.device)
        return batch_data

    def return_buffer(self, mode: str):
        assert mode in ['real', 'fake'], print(mode)
        return self.clean_buffer if mode == 'real' else self.attack_buffer

    def sample(self, mode: str, batch_size: int):
        buffer = self.return_buffer(mode)
        return buffer.sample(batch_size)

    def sample_state(self, mode: str, batch_size: int):
        buffer = self.return_buffer(mode)
        return buffer.sample_state(batch_size)

    def sample_all_data(self, mode: str):
        buffer = self.return_buffer(mode)
        return buffer.sample_all_data()

    def add_batch(self, mode: str, s: np.ndarray, a: np.ndarray,
                  r: np.ndarray, s2: np.ndarray, done: np.ndarray):
        # mode is in ['real', 'fake']
        buffer = self.return_buffer(mode)
        buffer.add_batch(s, a, r, s2, done)

# if __name__ == '__main__':
#     buf_cfg = {}
#     clean_dataset, attack_dataset = {}, {}
#     device = None
#     clean_buffer = hydra.utils.instantiate(buf_cfg)
#     attack_buffer = hydra.utils.instantiate(buf_cfg)
#     buffer = EntroBuffer(clean_dataset,
#                          attack_dataset,
#                          clean_buffer,
#                          attack_buffer,
#                          device)
#     batch_data = buffer.sample_data(real_ratio=0.5, batch_size=64)
