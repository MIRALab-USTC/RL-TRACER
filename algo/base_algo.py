import abc
import copy
import torch
import time
from pathlib import Path

from common.utils import log


class BaseAlgorithm(object, metaclass=abc.ABCMeta):

    def __init__(
        self, state_dim, action_dim, action_limit, actor_update_freq,
        critic_tau, critic_target_update_freq, device, gamma
    ):
        # Setting hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit = action_limit
        self.actor_update_freq = actor_update_freq
        self.critic_tau = critic_tau
        self.critic_target_update_freq = critic_target_update_freq
        self.device = device
        self.gamma = gamma
        self.training = False

        # Default hyperparameters
        self.reward_penalty = False
        self.reward_penalty_coef = 0.0
        self.pessimistic_func = None

        self.update_steps = 0
        self.update_actor_steps = 0
        self.update_critic_steps = 0
        self.total_time_steps = 0

        # Setting modules
        self.actor = None
        self.actor_targ = None
        self.critic = None
        self.critic_targ = None
        self.value = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.value_optimizer = None

    def train(self, training=True):
        self.training = training
        self.actor.train(training) if self.actor is not None else None
        self.critic.train(training) if self.critic is not None else None
        self.value.train(training) if self.value is not None else None
        self._train()

    def _train(self):
        pass

    def train_targ(self):
        self.actor_targ.train(self.training) if self.actor_targ is not None else None
        self.critic_targ.train(self.training) if self.critic_targ is not None else None
        self._train_targ()

    def _train_targ(self):
        pass

    def print_module(self):
        print("Actor:", self.actor) if self.actor is not None else None
        print("Critic:", self.critic) if self.critic is not None else None
        print("Value:", self.value) if self.value is not None else None
        self._print_module()

    def _print_module(self):
        pass

    def select_action(self, obs, deterministic=False, to_numpy=True):
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return self.actor.act(obs, deterministic, to_numpy)

    def estimate_q_val(self, obs, act, minimize=True):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
        if act.ndim == 1:
            act = act.unsqueeze(0)
        q_vals = self.critic(obs, act, minimize)
        return q_vals

    @abc.abstractmethod
    def update_critic(self, data, logger):
        raise NotImplementedError

    @abc.abstractmethod
    def update_actor(self, data, logger):
        raise NotImplementedError

    def update(self, fake_env, logger, save_log=False):
        data = fake_env.sample_data()
        q_info_dict, pi_info_dict = self._update(
            data, logger, self.total_time_steps, save_log=False)

        # save log
        if save_log:
            log(logger, 'train_critic', q_info_dict, self.total_time_steps)
            log(logger, 'train_actor', pi_info_dict, self.total_time_steps)

        q_info_dict.update(pi_info_dict)
        for k, v in q_info_dict.items():
            if not isinstance(v, float):
                q_info_dict[k] = v.mean().item()
        return q_info_dict

    @abc.abstractmethod
    def _update(self, data, logger, step, save_log=False):
        raise NotImplementedError

    def save(self, model_dir, step):
        if self.actor is not None:
            torch.save(self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step))
        if self.critic is not None:
            torch.save(self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step))
        if self.value is not None:
            torch.save(self.value.state_dict(), '%s/value_%s.pt' % (model_dir, step))
        self._save(model_dir, step)

    @abc.abstractmethod
    def _save(self, model_dir, step):
        raise NotImplementedError

    def load(self, model_dir, step):
        if step == -1:
            model_dir = Path(model_dir)
            assert model_dir.is_dir(), print('is_dir: ', model_dir.is_dir())
            dir_list = [i for i in Path.iterdir(model_dir)
                        if 'actor_' in str(i) and 'optim' not in str(i)]
            step = int(len(dir_list) * 5)

        if self.actor is not None:
            self.actor.load_state_dict(torch.load('%s/actor_%s.pt' % (model_dir, step),
                                                  map_location=self.device))
        if self.actor_targ is not None:
            self.actor_targ = copy.deepcopy(self.actor)

        if self.critic is not None:
            self.critic.load_state_dict(torch.load('%s/critic_%s.pt' % (model_dir, step),
                                                   map_location=self.device))
        if self.critic_targ is not None:
            self.critic_targ = copy.deepcopy(self.critic)

        if self.value is not None:
            self.value.load_state_dict(torch.load('%s/value_%s.pt' % (model_dir, step),
                                                  map_location=self.device))

        self._load(model_dir, step)
        print("| Load the Agent in Epoch %d." % step)

    @abc.abstractmethod
    def _load(self, model_dir, step):
        raise NotImplementedError

    def print_log(self, logger, epoch, env_type, start_time, epoch_fps):
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('TotalEnvInteracts', epoch * 1000)
        logger.log_tabular('TestEpRet', average_only=True)
        logger.log_tabular('TestScore', average_only=True)
        if env_type != '':
            logger.log_tabular(f'TestEpRet{env_type}', average_only=True)
            logger.log_tabular(f'TestScore{env_type}', average_only=True)
        logger.log_tabular('Qvals', average_only=True)
        logger.log_tabular('TQvals', average_only=True)
        logger.log_tabular('LossQ', average_only=True)
        logger.log_tabular('LossPi', average_only=True)

        self._print_log(logger)

        logger.log_tabular('Time', (time.time() - start_time)/3600)
        logger.log_tabular('FPS', epoch_fps)
        logger.dump_tabular()

    @abc.abstractmethod
    def _print_log(logger):
        raise NotImplementedError
