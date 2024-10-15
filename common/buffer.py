import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List, Optional, Tuple, Union


class UnionReplayBuffer:
    """
    Save the Data without Normalization
    """
    def __init__(
        self,
        capacity: int = 1000000,
        batch_size: int = 256,
        val_ratio: float = 0.2,
        num_workers: int = 8,
        state_dim: int = 11,
        action_dim: int = 3,
        reward_dim: int = 1,
        device: Any = None,
    ):
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, reward_dim), dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.terminals = np.empty((capacity, 1), dtype=np.float32)
        self.timestamp = np.empty((capacity, 1), dtype=np.float32)
        self.attack_indexes = None

        self.capacity = capacity
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.device = device if device.type == 'cuda' else torch.device('cpu')
        self.ptr = 0
        self.size = 0

    def load_dataset(self, state, action, reward, next_state, terminal, timestamp=None):
        self.states = state
        self.actions = action
        self.rewards = reward.reshape(-1, 1) if reward.ndim == 1 else reward
        self.next_states = next_state
        self.terminals = terminal.reshape(-1, 1) if terminal.ndim == 1 else terminal
        if timestamp is not None:
            self.timestamp = timestamp
        self.ptr = len(self.states)
        self.size = len(self.states)

    def add_attack_indexes(self, attack_indexes):
        self.attack_indexes = attack_indexes

    def add(self, state, act, rew, next_state, terminal):
        self.states[self.ptr] = np.array(state).copy()
        self.next_states[self.ptr] = np.array(next_state).copy()
        self.actions[self.ptr] = np.array(act).copy()
        self.rewards[self.ptr] = np.array(rew).copy()
        self.terminals[self.ptr] = np.array(terminal).copy()

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, state, act, rew, next_state, terminal):
        size = state.shape[0]
        if size + self.ptr < self.capacity:
            self.states[self.ptr:self.ptr+size] = np.array(state).copy()
            self.next_states[self.ptr:self.ptr+size] = np.array(next_state).copy()
            self.actions[self.ptr:self.ptr+size] = np.array(act).copy()
            self.rewards[self.ptr:self.ptr+size] = np.array(rew).copy()
            self.terminals[self.ptr:self.ptr+size] = np.array(terminal).copy()
        else:
            temp = self.capacity - self.ptr
            self.states[self.ptr:] = np.array(state[:temp]).copy()
            self.next_states[self.ptr:] = np.array(next_state[:temp]).copy()
            self.actions[self.ptr:] = np.array(act[:temp]).copy()
            self.rewards[self.ptr:] = np.array(rew[:temp]).copy()
            self.terminals[self.ptr:] = np.array(terminal[:temp]).copy()

            self.states[:size-temp] = np.array(state[temp:]).copy()
            self.next_states[:size-temp] = np.array(next_state[temp:]).copy()
            self.actions[:size-temp] = np.array(act[temp:]).copy()
            self.rewards[:size-temp] = np.array(rew[temp:]).copy()
            self.terminals[:size-temp] = np.array(terminal[temp:]).copy()

        self.ptr = (self.ptr + size) % self.capacity
        self.size = min(self.size + size, self.capacity)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = dict(
            state=self.states[idxs],
            next_state=self.next_states[idxs],
            action=self.actions[idxs],
            reward=self.rewards[idxs],
            not_done=1.-self.terminals[idxs]
        )
        if self.attack_indexes is not None:
            batch['attack_indexes'] = self.attack_indexes[idxs]
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def sample_state(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        # do not select the terminal states
        idxs = np.random.choice((1.-self.terminals).nonzero()[0], size=self.batch_size)
        # idxs = np.random.randint(0, self.size, size=batch_size)
        state = self.states[idxs]
        state_timestamp = self.timestamp[idxs]
        return torch.as_tensor(state, dtype=torch.float32), \
                torch.as_tensor(state_timestamp, dtype=torch.long)

    def sample_all_data(self):
        data = dict(
            state=self.states,
            next_state=self.next_states,
            action=self.actions,
            reward=self.rewards,
            not_done=1.-self.terminals
        )
        if self.attack_indexes is not None:
            data['attack_indexes'] = self.attack_indexes
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def generate_dataloader(self, transforms, val_ratio=None, max_num=1000):
        valid = False if val_ratio is None and self.val_ratio == 0 else True
        val_ratio = self.val_ratio if val_ratio is None else val_ratio

        terminals = self.terminals.astype(bool).flatten()
        states = np.delete(self.states, terminals, axis=0)
        next_states = np.delete(self.next_states, terminals, axis=0)
        actions = np.delete(self.actions, terminals, axis=0)
        rewards = np.delete(self.rewards, terminals, axis=0)

        idxs = np.arange(states.shape[0])
        np.random.shuffle(idxs)
        temp = min(int(states.shape[0] * val_ratio), max_num)

        unnorm_train_buffer = {
            'state': states[idxs].copy()[temp:],
            'action': actions[idxs].copy()[temp:],
            'reward': rewards[idxs].copy()[temp:],
            'delta_state': next_states[idxs].copy()[temp:] - states[idxs].copy()[temp:],
        }
        unnorm_val_buffer = {
            'state': states[idxs].copy()[:temp],
            'action': actions[idxs].copy()[:temp],
            'reward': rewards[idxs].copy()[:temp],
            'delta_state': next_states[idxs].copy()[:temp] - states[idxs].copy()[:temp],
        } if valid else None

        train_buffer = transforms(**unnorm_train_buffer)
        val_buffer = transforms(**unnorm_val_buffer) if valid else None

        train_data = DataBuffer(train_buffer, self.batch_size, self.device)
        val_data = DataBuffer(val_buffer, self.batch_size, self.device) if valid else None

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, 
                                  num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True, 
                                num_workers=self.num_workers, pin_memory=True) if valid else None
        return train_loader, val_loader

    def __len__(self):
        return self.size


class DataBuffer(Dataset):
    """
    Save the Data with Normalization to Train Models
    """
    def __init__(self, buffer, batch_size, device):
        super().__init__()
        self.buffer = buffer
        self.batch_size = batch_size
        self.device = device

    def __len__(self):
        return len(self.buffer[0])

    def __getitem__(self, idx):
        state, act, rew, delta_state = self.buffer
        data = dict(
            state=state[idx],
            delta_state=delta_state[idx],
            act=act[idx],
            rew=rew[idx],
        )
        data = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
        input = torch.cat([data['state'], data['act']], dim=-1)
        label = torch.cat([data['rew'], data['delta_state']], dim=-1)
        return input, label
