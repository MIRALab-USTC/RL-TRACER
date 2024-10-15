import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from common import utils

LOG_STD_MAX_VALUES = 2.0
LOG_SIG_MIN_VALUES = -5.0


class VectorizedGaussianCritic(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 l: int = 2, num_q: int = 5, output_dim: int = 1, activation: str = "relu",
                 q_std: List[float] = [-5.0, 2.0], std_architecture: str = "weight"):
        super().__init__()
        self.output_dim = output_dim
        self.num_q = num_q
        self.hidden_depth = l
        self.std_architecture = std_architecture
        self.Q_LOG_STD_MIN, self.Q_LOG_STD_MAX = q_std

        self.q_trunk = utils.MLP([state_dim + action_dim, *([hidden_dim] * l)],
                                 activation_fn=activation,
                                 output_activation_fn=activation,
                                 ensemble_size=num_q)
        self.q_mean = utils.MLP([hidden_dim, output_dim],
                                ensemble_size=num_q)
        if std_architecture == "weight":
            self.q_log_std = nn.Parameter(torch.zeros(
                    num_q, output_dim, dtype=torch.float32))
        elif std_architecture == "mlp":
            self.q_log_std = utils.MLP([hidden_dim, output_dim],
                                       ensemble_size=num_q)
        else:
            raise NotImplementedError

        # init as in the EDAC paper
        self.apply(utils.edac_init)

        torch.nn.init.uniform_(self.q_mean[0].weight, -3.e-3, 3.e-3)
        torch.nn.init.uniform_(self.q_mean[0].bias, -3.e-3, 3.e-3)
        self.infos = dict()

    def q(self, state_action: torch.Tensor) -> Tuple[torch.Tensor]:
        # [num_q, batch_size, hidden_dim]
        h = self.q_trunk(state_action)
        # [num_q, batch_size, 1]
        q_mean = self.q_mean(h)

        if self.std_architecture == "weight":
            # [num_q, 1]
            q_std = torch.exp(self.q_log_std.clamp(self.Q_LOG_STD_MIN,
                                                   self.Q_LOG_STD_MAX))
            # [num_q, batch_size, 1]
            q_std = q_std.repeat_interleave(q_mean.size(-1), dim=-1)
        elif self.std_architecture == "mlp":
            # [num_q, batch_size, 1]
            q_log_std = self.q_log_std(h)
            q_std = torch.exp(q_log_std.clamp(self.Q_LOG_STD_MIN,
                                              self.Q_LOG_STD_MAX))
        else:
            raise NotImplementedError

        self.infos['q_mu'] = q_mean.squeeze(-1)
        self.infos['q_std'] = q_std.squeeze(-1)
        return q_mean.view(-1, 1), q_std.view(-1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # [num_q, batch_size, state_dim + action_dim]
        state_action = inputs.unsqueeze(0).repeat_interleave(self.num_q, dim=0)
        # [num_q * batch_size, 1]
        q_mean, q_std = self.q(state_action)
        # [num_q * batch_size, 1, 1]
        covariance_matrix  = torch.diag_embed(q_std, offset=0, dim1=1)
        return MultivariateNormal(q_mean, covariance_matrix)

    def log(self, L, step, log_freq, param=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        if param:
            for i in range(self.hidden_depth+1):
                L.log_param('train_critic/ensemble_q_fc%d_param' % i, self.q_trunk[i * 2], step)
            if self.std_architecture == "mlp":
                L.log_param('train_critic/ensemble_q_fc%d_param' % (self.hidden_depth + 1),
                            self.q_log_std[0], step)


class DistributionalCritic(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 l: int = 2, num_q: int = 5, cosines_dim: int = 256, output_dim: int = 1,
                 activation: str = "relu", init_type: str = "edac"):
        super().__init__()
        self.output_dim = output_dim
        self.cosines_dim = cosines_dim
        self.num_q = num_q
        self.hidden_depth = l

        self.q = QuantQFunction(input_dim = state_dim + action_dim,
                                hidden_dim=hidden_dim,
                                hidden_depth = l,
                                cosines_dim = cosines_dim,
                                num_q = num_q,
                                output_dim = output_dim,
                                activation = activation,
                                init_type = init_type)
        self.range_pi = torch.arange(start = 1, end =  cosines_dim + 1,
                                     dtype = torch.float32) * np.pi

        self.infos = dict()

    def forward(self, inputs: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        tau:    [num_q, batch_size, num_quantiles]
        """
        # [num_q, batch_size, state_dim + action_dim]
        state_action = inputs.unsqueeze(0).repeat_interleave(self.num_q, dim=0)

        if self.range_pi.device != tau.device:
            self.range_pi = self.range_pi.to(tau.device)

        # [num_q, batch_size, num_quantiles, cosines_dim]
        cosines = torch.cos(tau.unsqueeze(-1) * self.range_pi)

        # [num_q, batch_size, num_quantiles]
        q_quant = self.q(state_action, cosines)

        for i in range(q_quant.size(0)):
            self.infos['q%s' % (i + 1)] = q_quant[i]
        return q_quant

    def log(self, L, step, log_freq, param=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        self.q.log(L, step, log_freq, param)


class QuantQFunction(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_depth: int,
                 cosines_dim: int, num_q: int, output_dim: int,
                 activation: str, init_type: str) -> None:
        super().__init__()
        self.hidden_depth = hidden_depth
        self.q_trunk = utils.MLP([input_dim, hidden_dim, hidden_dim],
                                 activation_fn=activation,
                                 ensemble_size=num_q)
        self.q_tau = utils.MLP([cosines_dim, hidden_dim],
                               output_activation_fn=nn.ReLU,
                               ensemble_size=num_q)
        self.q_val = utils.MLP([*([hidden_dim] * hidden_depth), output_dim],
                                activation_fn=activation,
                                squeeze_output=True,
                                ensemble_size=num_q)
        getattr(self, f"{init_type}_init")()

    def forward(self, state_action: torch.Tensor, cosines: torch.Tensor) -> torch.Tensor:
        """
        cosines:    [num_q, batch_size, num_quantiles, cosines_dim]
        """
        # [num_q, batch_size, 1, hidden_dim]
        q = self.q_trunk(state_action).unsqueeze(-2)
        # [num_q, batch_size, num_quantiles, hidden_dim]
        tau = self.q_tau(cosines)
        # [num_q, batch_size, num_quantiles, hidden_dim]
        q = torch.mul(tau, q)
        # [num_q, batch_size, num_quantiles]
        q_quant = self.q_val(q)
        return q_quant

    def log(self, L, step, log_freq, param=False):
        if not log_freq or step % log_freq != 0:
            return

        if param:
            L.log_param('train_critic/ensemble_q_fc-1_param', self.q_tau[0], step)
            L.log_param('train_critic/ensemble_q_fc0_param', self.q_trunk[0], step)
            L.log_param('train_critic/ensemble_q_fc1_param', self.q_trunk[2], step)
            for i in range(self.hidden_depth):
                L.log_param('train_critic/ensemble_q_fc%d_param' % (i+2), self.q_val[i * 2], step)

    def edac_init(self):
        # init as in the EDAC paper
        self.apply(utils.edac_init)
        torch.nn.init.uniform_(self.q_val[-2].weight, -3.e-3, 3.e-3)
        torch.nn.init.uniform_(self.q_val[-2].bias, -3.e-3, 3.e-3)

    def fanin_init(self):
        self.apply(utils.fanin_init)

    def xavier_init(self):
        self.apply(utils.weight_init)

    def kaiming_init(self):
        self.apply(utils.weight_init_v2)

    def truncated_init(self):
        self.apply(utils.weight_init_v3)


class DistributionalValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, l: int, cosines_dim: int,
                 output_dim: int = 1, activation: str = "relu", init_type: str = "edac") -> None:
        super().__init__()
        self.hidden_depth = l
        dims = [state_dim, *([hidden_dim] * l), 1]
        self.v = utils.MLP(dims, squeeze_output = True)
        self.v_trunk = utils.MLP([state_dim, hidden_dim, hidden_dim],
                                 activation_fn = activation)
        self.v_tau = utils.MLP([cosines_dim, hidden_dim],
                               output_activation_fn = nn.ReLU)
        self.v_val = utils.MLP([*([hidden_dim] * l), output_dim],
                                activation_fn = activation,
                                squeeze_output = True)
        self.range_pi = torch.arange(start = 1, end =  cosines_dim + 1,
                                     dtype = torch.float32) * np.pi

        getattr(self, f"{init_type}_init")()
        self.infos = dict()

    def forward(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        tau:    [batch_size, num_quantiles]
        """
        # [batch_size, state_dim]
        v = self.v_trunk(state)

        if self.range_pi.device != tau.device:
            self.range_pi = self.range_pi.to(tau.device)
        # [batch_size, num_quantiles, cosines_dim]
        cosines = torch.cos(tau.unsqueeze(-1) * self.range_pi)

        # [batch_size, 1, hidden_dim]
        v = self.v_trunk(state).unsqueeze(-2)
        # [batch_size, num_quantiles, hidden_dim]
        tau = self.v_tau(cosines)
        # [batch_size, num_quantiles, hidden_dim]
        v = torch.mul(tau, v)
        # [batch_size, num_quantiles]
        v_quant = self.v_val(v)

        self.infos['v'] = v_quant
        return v_quant

    def log(self, L, step, log_freq, param=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_value/%s_hist' % k, v, step)

        if param:
            L.log_param('train_critic/v_fc-1_param', self.v_tau[0], step)
            L.log_param('train_critic/v_fc0_param', self.v_trunk[0], step)
            L.log_param('train_critic/v_fc1_param', self.v_trunk[2], step)
            for i in range(self.hidden_depth):
                L.log_param('train_critic/v_fc%d_param' % (i+2), self.v_val[i * 2], step)

    def edac_init(self):
        # init as in the EDAC paper
        self.apply(utils.edac_init)
        torch.nn.init.uniform_(self.v_val[-2].weight, -3.e-3, 3.e-3)
        torch.nn.init.uniform_(self.v_val[-2].bias, -3.e-3, 3.e-3)

    def fanin_init(self):
        self.apply(utils.fanin_init)

    def xavier_init(self):
        self.apply(utils.weight_init)

    def kaiming_init(self):
        self.apply(utils.weight_init_v2)

    def truncated_init(self):
        self.apply(utils.weight_init_v3)


class ObservationModel(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, reward_dim: int,
                 q_dim: int, hidden_dim: int, l: int = 2, num_model: int = 1,
                 sigma: float = 0.3, model_type: str = "gaussian",
                 activation: str = "relu", std_architecture: str = "mlp"):
        super().__init__()
        input_dim = state_dim * 2 + action_dim + reward_dim + q_dim
        output_dim = state_dim + action_dim + reward_dim

        self.trunk = utils.MLP([input_dim, *([hidden_dim] * l)],
                               activation_fn=activation,
                               output_activation_fn=activation,
                               ensemble_size=num_model if num_model > 1 else None)
        self.mean = utils.MLP([hidden_dim, output_dim],
                              ensemble_size=num_model if num_model > 1 else None)
        self.log_std = None
        if model_type == "gaussian":
            if std_architecture == "mlp":
                self.log_std = utils.MLP([hidden_dim, output_dim],
                                         ensemble_size=num_model if num_model > 1 else None)
            elif std_architecture == "weight":
                self.log_std = nn.Parameter(torch.zeros(output_dim, dtype=torch.float32))
            else:
                raise NotImplementedError('no type of std architecture: %s' % std_architecture)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.hidden_depth = l
        self.num_model = num_model
        self.sigma = sigma
        self.model_type = model_type
        self.std_architecture = std_architecture
        self.total_losses = np.zeros((3, int(num_model // 3)))
        # init mask matrix
        self._init_mask_matrix(state_dim, action_dim, reward_dim,
                               state_dim * 2 + action_dim + reward_dim,
                               output_dim)

        self.infos = dict()

    def _init_mask_matrix(self, state_dim: int, action_dim: int, reward_dim: int,
                          input_dim: int, output_dim: int):
        num_model = int(self.num_model // 3)
        # [a, s', r], [s, s', r], [s, a, s']
        input_mask_index = [[0, state_dim]] * num_model + \
                [[state_dim, state_dim + action_dim]] * num_model + \
                [[state_dim * 2 + action_dim, input_dim]] * num_model
        input_mask_matrix = torch.ones((self.num_model, input_dim))
        for i, index in enumerate(input_mask_index):
            input_mask_matrix[i, index[0]:index[1]] = 0
        self.input_mask_matrix = input_mask_matrix.bool()

        valid_dim = self.num_model * input_dim - num_model * (
                state_dim + action_dim + reward_dim)
        assert self.input_mask_matrix.sum() == valid_dim, \
                print(self.input_mask_matrix.sum())

        # [s], [a], [r]
        output_mask_index = [[0, state_dim]] * num_model + \
                [[state_dim, state_dim + action_dim]] * num_model + \
                [[state_dim + action_dim, output_dim]] * num_model
        output_mask_matrix = torch.zeros((self.num_model, output_dim))
        for i, index in enumerate(output_mask_index):
            output_mask_matrix[i, index[0]:index[1]] = 1
        self.output_mask_matrix = output_mask_matrix.bool()

        valid_dim = num_model * (state_dim + action_dim + reward_dim)
        assert self.output_mask_matrix.sum() == valid_dim, \
                print(self.output_mask_matrix.sum())

    def to(self, device):
        super(ObservationModel, self).to(device)
        self.input_mask_matrix = self.input_mask_matrix.to(device)
        self.output_mask_matrix = self.output_mask_matrix.to(device)
        return self

    def compute_log_std(self, h: torch.Tensor) -> torch.Tensor:
        '''
        h: [num_model, batch_size, hidden_dim]
        '''
        if self.log_std is None:
            return None

        if self.model_type == "gaussian":

            if self.std_architecture == "mlp":
                # [num_model, batch_size, output_dim]
                log_std = self.log_std(h).clamp(LOG_SIG_MIN_VALUES, LOG_STD_MAX_VALUES)
                self.infos['std'] = torch.exp(log_std)

            elif self.std_architecture == "weight":
                log_std = self.log_std.clamp(LOG_SIG_MIN_VALUES, LOG_STD_MAX_VALUES)
                # [num_model, batch_size, output_dim]
                log_std = log_std.view(1, 1, -1).repeat(h.size(0), h.size(1), 1)

                for i in range(h.size(0)):
                    self.infos['std%s' % (i + 1)] = torch.exp(log_std)[i]
            else:
                raise NotImplementedError('no type of std architecture: %s'
                                          % self.std_architecture)
            return log_std
        raise NotImplementedError('no type of model_type: %s' % self.model_type)

    def forward(self, inputs: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        '''
        inputs: [s, a, s', r]
        outputs: [s, a, r]
        '''
        # [num_model, batch_size, input_dim]
        inputs = inputs.unsqueeze(0).repeat_interleave(self.num_model, dim=0) \
                * self.input_mask_matrix.unsqueeze(1)
        # [num_model, batch_size, input_dim + num_quantiles]
        obs = torch.cat([inputs, d.unsqueeze(0).repeat_interleave(self.num_model, dim=0)],
                        dim=-1)
        # [num_model, batch_size, hidden_dim]
        h = self.trunk(obs)
        # [num_model, batch_size, output_dim]
        mean = self.mean(h)
        # [num_model, batch_size, output_dim]
        log_std = self.compute_log_std(h)
        for i in range(mean.size(0)):
            self.infos['mu%s' % (i + 1)] = mean[i]

        mean *= self.output_mask_matrix.unsqueeze(1)
        if log_std is not None:
            log_std *= self.output_mask_matrix.unsqueeze(1)
            assert not log_std.isnan().any(), print('log_std has nan. inputs has nan? ', inputs.isnan().any())
        return mean, log_std

    def calc_loss(self, mean: torch.Tensor, log_std: Optional[torch.Tensor],
                  label: torch.Tensor) -> torch.Tensor:
        num_model = mean.size(0)
        # [num_model, batch_size, output_dim]
        label = label.unsqueeze(0).repeat_interleave(num_model, dim=0)
        label = (label * self.output_mask_matrix.unsqueeze(1)).detach()

        if self.model_type != "gaussian" or log_std is None:
            assert isinstance(mean, torch.Tensor), print(mean)
            loss = utils.huber_loss(mean - label, sigma=self.sigma).mean(1).sum(-1)
        else:
            inv_std = torch.exp(-log_std)
            assert inv_std.ndim < 4, print(inv_std.shape)
            if inv_std.ndim == 1:
                inv_std = inv_std.view(1, 1, -1)
            elif inv_std.ndim == 2:
                inv_std = inv_std.unsqueeze(0)
            # [num_model, batch_size, output_dim]
            # loss_recon_mse = (((mean - label) / inv_std).pow(2)).mean(1).sum(-1)
            loss_recon_mse = utils.huber_loss((mean - label) / inv_std,
                                              sigma=self.sigma).mean(1).sum(-1)
            loss = loss_recon_mse + (log_std * 2).mean(1).sum(-1)

        # equal to compute .mean()
        loss *= (num_model / self.output_mask_matrix.sum())
        self.total_losses += loss.detach().cpu().data.numpy().reshape(3, -1)
        return loss

    def dist(self, mean: torch.Tensor, std: torch.Tensor) -> MultivariateNormal:
        '''
        mean: [num_model, batch_size, output_dim]
        std:  [num_model, batch_size, output_dim]
        '''
        # [num_model, batch_size, output_dim, output_dim]
        covariance_matrix  = torch.diag_embed(std, offset=0, dim1=std.ndim-1)
        # [num_model * batch_size, output_dim, output_dim]
        if covariance_matrix.ndim == 3:
            covariance_matrix = covariance_matrix.repeat(self.num_model, 1, 1)
        elif covariance_matrix.ndim == 4:
            covariance_matrix = covariance_matrix.view(-1, *covariance_matrix.shape[-2:])
        else:
            raise ValueError(covariance_matrix.shape)
        return MultivariateNormal(mean.view(-1, mean.size(-1)), covariance_matrix)

    def argmin_index(self):
        num_model, l = self.total_losses.shape
        argmin = self.total_losses.argmin(-1)
        argmin_index = np.array([argmin[i] + i * l for i in range(num_model)])
        # [3,]
        return argmin_index

    def get_state(self, samples: torch.Tensor) -> torch.Tensor:
        batch_size = samples.shape[1]
        # [3, num_model // 3, output_dim]
        output_mask_matrix = self.output_mask_matrix.reshape(3, self.num_model//3, -1)
        state = torch.masked_select(
                samples[0], output_mask_matrix[0][0].bool()).view(batch_size, -1)
        assert state.size(-1) == self.state_dim, print(state.shape)
        return state

    def get_action(self, samples: torch.Tensor) -> torch.Tensor:
        batch_size = samples.shape[1]
        # [3, num_model // 3, output_dim]
        output_mask_matrix = self.output_mask_matrix.reshape(3, self.num_model//3, -1)
        # [batch_size, state_dim]
        action = torch.masked_select(
                samples[1], output_mask_matrix[1][0].bool()).view(batch_size, -1)
        assert action.size(-1) == self.action_dim, print(action.shape)
        return action

    def get_reward(self, samples: torch.Tensor) -> torch.Tensor:
        batch_size = samples.shape[1]
        # [3, num_model // 3, output_dim]
        output_mask_matrix = self.output_mask_matrix.reshape(3, self.num_model//3, -1)
        # [batch_size, state_dim]
        reward = torch.masked_select(
                samples[2], output_mask_matrix[2][0].bool()).view(batch_size, -1)
        assert reward.size(-1) == self.reward_dim, print(reward.shape)
        return reward

    def get_data(self, samples: torch.Tensor) -> Tuple[torch.Tensor]:
        batch_size = samples.shape[1]
        # [3, num_model // 3, output_dim]
        output_mask_matrix = self.output_mask_matrix.reshape(3, self.num_model//3, -1)
        # [batch_size, state_dim]
        state = torch.masked_select(
                samples[0], output_mask_matrix[0][0].bool()).view(batch_size, -1)
        # [batch_size, action_dim]
        action = torch.masked_select(
                samples[1], output_mask_matrix[1][0].bool()).view(batch_size, -1)
        # [batch_size, reward_dim]
        reward = torch.masked_select(
                samples[2], output_mask_matrix[2][0].bool()).view(batch_size, -1)

        assert state.size(-1) == self.state_dim and \
                action.size(-1) == self.action_dim and \
                reward.size(-1) == self.reward_dim, \
                print(state.shape, action.shape, reward.shape)
        return state, action, reward

    def sample(self, mean: torch.Tensor, log_std: Optional[torch.Tensor]):
        '''
        mean: [num_model, batch_size, output_dim]
        log_std: [num_model, batch_size, output_dim]
        samples: [3, batch_size, output_dim]
        '''
        if self.model_type != "gaussian" or log_std is None:
            # [batch_size, output_dim], sample the one with minimum total loss
            samples = mean
        else:
            # [num_model, batch_size, output_dim]
            samples = mean + torch.randn_like(mean) * torch.exp(log_std)
            # dist = self.dist(mean, torch.exp(log_std))
            # samples = dist.rsample().view(*mean.shape)
            # [batch_size, output_dim], sample the one with minimum total loss
        if samples.size(0) == 3:
            return self.get_data(samples)
        return self.get_data(samples[self.argmin_index()])

    def log(self, L, step, log_freq, param=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_model/%s_hist' % k, v, step)

        if param:
            for i in range(self.hidden_depth+1):
                L.log_param('train_model/model_fc%d_param' % i, self.trunk[i * 2], step)
            L.log_param('train_model/mean_param', self.mean[0], step)
            L.log_param('train_model/log_std_param', self.log_std.detach(), step)
            if self.log_std is not None:
                if self.model_type == "gaussian":
                    if self.std_architecture == "mlp":
                        L.log_param('train_model/log_std_param', self.log_std[0], step)
                    elif self.std_architecture == "weight":
                        L.log_param('train_model/log_std_param', self.log_std.detach(), step)
                    else:
                        raise NotImplementedError('no type of std architecture: %s' %
                                                  self.std_architecture)


class InverseModel(nn.Module):
    def __init__(self, state_dim: int, quantile_dim: int, action_dim: int,
                 hidden_dim: int = 256, l: int = 2, act_limit: float = 1.0,
                 num_model: int = 1, model_type: str = "gaussian"):
        super().__init__()
        dims = [state_dim * 2 + quantile_dim, *([hidden_dim] * l), action_dim]
        self.pi = utils.MLP(dims, ensemble_size=num_model if num_model > 1 else None)
        self.log_std = None
        if model_type == "gaussian":
            self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

        self.max_action = act_limit
        self.hidden_depth = l
        self.num_model = num_model
        self.model_type = model_type
        self.total_losses = np.zeros(num_model)
        self.infos = dict()

    def dist(self, mean: torch.Tensor, std: torch.Tensor) -> MultivariateNormal:
        '''
        mean: [num_model, batch_size, action_dim]
        std:  [batch_size, action_dim]
        '''
        # [batch_size, action_dim, action_dim]
        covariance_matrix  = torch.diag_embed(std, offset=0, dim1=1)
        # [num_model * batch_size, action_dim, action_dim]
        covariance_matrix = covariance_matrix.repeat(self.num_model, 1, 1)
        return MultivariateNormal(mean.view(-1, mean.size(-1)), covariance_matrix)

    def forward(self, s: torch.Tensor, s2: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # [num_model, batch_size, state_dim * 2 + num_quantiles]
        obs = torch.cat([s, s2, d],
                        dim=-1).unsqueeze(0).repeat_interleave(self.num_model, dim=0)
        # [num_model, batch_size, action_dim]
        mean = self.pi(obs)
        mean_clamp = torch.clamp(self.max_action * mean,
                                 -self.max_action,
                                 self.max_action).detach()
        mean = mean - mean.detach() + mean_clamp

        for i in range(mean.size(0)):
            self.infos['mu%s' % (i + 1)] = mean[i]
        return mean

    def calc_loss(self, act: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # [num_model, batch_size, action_dim]
        label = label.unsqueeze(0).repeat_interleave(act.size(0), dim=0)

        if self.model_type != "gaussian":
            assert isinstance(act, torch.Tensor), print(act)
            loss = utils.huber_loss(act - label, sigma=0.3).mean([1, 2])

            self.total_losses += loss.detach().cpu().data.numpy().flatten()
            return loss

        assert self.log_std is not None, print(self.model_type)
        # [batch_size, action_dim]
        log_std = self.log_std.clamp(LOG_SIG_MIN_VALUES, LOG_STD_MAX_VALUES)
        self.infos['std'] = torch.exp(log_std).detach()

        inv_var = torch.exp(-log_std * 2).view(1, 1, -1)
        # [num_model, batch_size, action_dim]
        loss_recon_mse = ((act - label).pow(2) * inv_var)
        loss = loss_recon_mse.mean([1, 2]) + (log_std * 2).mean()

        self.total_losses += loss.detach().cpu().data.numpy().flatten()
        return loss

    def act(self, a_mean: torch.Tensor):
        '''
        a_mean: [num_model, batch_size, action_dim]
        '''
        if self.model_type != "gaussian":
            # [batch_size, action_dim], sample the one with minimum total loss
            return a_mean[self.total_losses.argmin()]

        log_std = self.log_std.clamp(LOG_SIG_MIN_VALUES, LOG_STD_MAX_VALUES)
        # [batch_size, action_dim]
        log_std = log_std.unsqueeze(0).repeat_interleave(a_mean.size(1), dim=0)
        dist = self.dist(a_mean, torch.exp(log_std))
        # [num_model, batch_size, action_dim]
        a_samples = dist.rsample().view(*a_mean.shape)
        # [batch_size, action_dim], sample the one with minimum total loss
        return a_samples[self.total_losses.argmin()]

    def log(self, L, step, log_freq, param=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        if param:
            for i in range(self.hidden_depth+1):
                L.log_param('train_actor/pi_fc%d_param' % i, self.pi[i * 2], step)



