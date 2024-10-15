import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from common.distributions import TanhNormal

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from common import utils


LOG_STD_MAX = 2
LOG_STD_MIN = -20
MEAN_MIN = -6.0
MEAN_MAX = 2.0
LOG_STD_MAX_VALUES = 2.0
LOG_SIG_MIN_VALUES = -5.0


class SGMLPActor(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 1024,
                 l: int = 2, act_limit: float = 1.0, activation: str = 'relu'):
        super(SGMLPActor, self).__init__()
        self.act_limit = act_limit
        self.hidden_depth = l
        self.trunk = utils.MLP([state_dim, *([hidden_dim] * l)],
                               output_activation_fn=activation)

        self.mean = utils.MLP([hidden_dim, action_dim])
        self.log_std = utils.MLP([hidden_dim, action_dim])
        self.infos = dict()
        self.apply(utils.weight_init)

    def _reprocess(self, pi):
        pi[pi == 1.0] -= 1e-10
        pi[pi == -1.0] += 1e-10
        return pi

    def _output(self, pi):
        if pi is None:
            return None
        return self._reprocess(self.act_limit * pi)

    def dist(self, state):
        # mu, log_std = self.pi(state).chunk(2, dim=-1)
        h = self.trunk(state)
        mu = self.mean(h)
        log_std = self.log_std(h)

        # constrain log_std inside [LOG_STD_MIN, LOG_STD_MAX]
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        self.infos['pre_mu'] = mu
        self.infos['std'] = log_std.exp()
        return mu, log_std

    def forward(self, state, compute_pi=True, with_logprob=True):
        mu, log_std = self.dist(state)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
            self.infos['pre_act'] = pi
        else:
            pi = None

        if with_logprob:
            log_pi = utils.gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = utils.squash(mu, pi, log_pi)
        self.infos['mu'] = mu
        if pi is not None:
            self.infos['act'] = self._output(pi)
        return self._output(mu), self._output(pi), log_pi, log_std

    def log_prob(self, state, action):
        mu, log_std = self.dist(state)
        mu = torch.clamp(mu, MEAN_MIN, MEAN_MAX)

        raw_action = utils.atanh(action)
        inv_std = torch.exp(-log_std)
        noise = (raw_action - mu) * inv_std

        log_pi = utils.gaussian_logprob(noise, log_std)
        log_pi -= torch.log(F.relu(1 - action.pow(2)) + 1e-6).sum(-1, keepdim=True)
        return log_pi.squeeze(-1)

    def act(self, state, deterministic=False, to_numpy=True):
        mu_action, pi_action, _, _ = self.forward(state, not deterministic, False)
        action = mu_action if deterministic else pi_action

        return action.squeeze(0).cpu().detach().numpy() if to_numpy \
            else action.squeeze(0).detach()

    def log(self, L, step, log_freq, param=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        if param:
            # for i in range(self.hidden_depth+1):
                # L.log_param('train_actor/pi_fc%d' % i, self.pi[i * 2], step)
            for i in range(self.hidden_depth):
                L.log_param('train_actor/fc%d_param' % i, self.trunk[i * 2], step)
            L.log_param('train_actor/mean_param', self.mean[0], step)
            L.log_param('train_actor/log_std_param', self.log_std[0], step)


class MLPActor(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 1024,
                 l: int = 2, act_limit: float = 1.0, act_noise: float = 0.1,
                 activation: str = 'tanh'):
        super(MLPActor, self).__init__()
        self.act_limit = act_limit
        self.act_noise = act_noise
        self.hidden_depth = l
        self.pi = utils.MLP([state_dim, *([hidden_dim] * l), action_dim],
                            output_activation_fn=activation)
        self.infos = dict()
        self.apply(utils.weight_init)

    def forward(self, state: torch.Tensor, deterministic: bool = False,
                act_noise: Optional[float] = None, clip: bool = False,
                noise_clip: float = 0.5) -> torch.Tensor:
        mu = self.act_limit * self.pi(state)
        self.infos['mu'] = mu

        if deterministic:
            pi_action = mu
        else:
            if act_noise is None: act_noise = self.act_noise
            self.infos['std'] = act_noise
            noise = torch.randn_like(mu) * act_noise
            if clip:
                noise = noise.clamp(-noise_clip, noise_clip)
            pi_action = (mu + noise).clamp(-self.act_limit, self.act_limit)
            self.infos['act'] = pi_action

        return pi_action

    def act(self, state: torch.Tensor, deterministic: bool = False, to_numpy: bool = True):
        action = self.forward(state, deterministic)
        return action.squeeze(0).cpu().detach().numpy() if to_numpy \
            else action.squeeze(0).detach()

    def log(self, L, step, log_freq, param=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        if param:
            for i in range(self.hidden_depth+1):
                L.log_param('train_actor/pi_fc%d_param' % i, self.pi[i * 2], step)


class TanhGaussianActor(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 1024,
                 l: int = 2, act_limit: float = 1.0, activation: str = 'relu'):
        super(TanhGaussianActor, self).__init__()
        self.act_limit = act_limit
        self.hidden_depth = l

        self.trunk = utils.MLP([state_dim, *([hidden_dim] * (l - 1))],
                               activation_fn=activation,
                               output_activation_fn=activation)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32,
                                                requires_grad=True))
        self.infos = dict()
        self.apply(utils.weight_init)

    def _output(self, pi):
        if pi is None:
            return None
        return self.act_limit * pi

    def dist(self, state: torch.Tensor, with_mu_clamp=False) -> TanhNormal:
        # mu, log_std = self.pi(state).chunk(2, dim=-1)
        mu = self.mean(self.trunk(state))

        if with_mu_clamp:
            # constrain mu inside [MEAN_MIN, MEAN_MAX]
            mu = torch.clamp(mu, MEAN_MIN, MEAN_MAX)
        # std = torch.exp(LOG_SIG_MIN_VALUES + torch.sigmoid(self.log_std) * (
        #         LOG_STD_MAX_VALUES - LOG_SIG_MIN_VALUES))
        std = torch.exp(self.log_std.clamp(LOG_SIG_MIN_VALUES, LOG_STD_MAX_VALUES))

        self.infos['mu'] = mu
        self.infos['std'] = std
        return TanhNormal(mu, std)

    def forward(self, state: torch.Tensor, compute_act: bool = True,
                with_logprob: bool = True) -> Tuple[torch.Tensor]:
        dist = self.dist(state)
        act = None
        log_pi_act = None
        log_std = dist.stddev

        if compute_act:
            act, pre_tanh_value = dist.rsample(return_pre_tanh_value=True)
            if with_logprob:
                log_pi_act = dist.log_prob(
                        act, pre_tanh_value=pre_tanh_value)
                log_pi_act = log_pi_act.sum(dim=-1)
                # log_pi_act = log_pi.sum(dim=-1, keepdim=True)
            self.infos['act'] = act
        return self._output(dist.mean), self._output(act), log_pi_act, log_std

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raw_action = utils.atanh(action)
        dist = self.dist(state, with_mu_clamp=True)

        log_prob = dist.log_prob(value=action, pre_tanh_value=raw_action)
        return log_prob.sum(-1)

    def act(self, state: torch.Tensor, deterministic: bool = False, to_numpy: bool = True):
        mu_action, pi_action, _, _ = self(state, not deterministic, False)
        action = self._output(mu_action) if pi_action is None else self._output(pi_action)
        return action.cpu().detach().numpy().flatten() if to_numpy \
            else action.squeeze(0).detach()

    def log(self, L, step, log_freq):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        for i in range(self.hidden_depth):
            L.log_param('train_actor/pi_fc%d_param' % i, self.trunk[i * 2], step)
        L.log_param('train_actor/mean_param', self.mean, step)
        L.log_param('train_actor/log_std_param', self.log_std.detach(), step)


class GaussianActor(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 l: int = 2, act_limit: float = 1.0):
        super().__init__()
        self.pi = utils.MLP([state_dim, *([hidden_dim] * l), action_dim])
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.max_action = act_limit
        self.hidden_depth = l
        self.infos = dict()

    def forward(self, obs: torch.Tensor) -> MultivariateNormal:
        mean = self.pi(obs)
        std = torch.exp(self.log_std.clamp(LOG_SIG_MIN_VALUES, LOG_STD_MAX_VALUES))
        self.infos['mu'] = mean
        self.infos['std'] = std
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)

    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = False, to_numpy: bool = True):
        dist = self(state)
        action = dist.mean if deterministic else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten() if to_numpy \
                else action.squeeze(0).detach()

    def log(self, L, step, log_freq, param=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        if param:
            for i in range(self.hidden_depth+1):
                L.log_param('train_actor/pi_fc%d_param' % i, self.pi[i * 2], step)


class Critic(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 l: int = 2, output_mod: Callable[[], nn.Module] = None,
                 num_q: int = 2, output_dim: int = 1, activation: str = 'relu'):
        super(Critic, self).__init__()
        self.output_dim = output_dim
        self.hidden_depth = l
        dims = [state_dim + action_dim, *([hidden_dim] * l), output_dim]
        self.q1 = utils.MLP(dims, activation_fn=activation,
                            output_activation_fn=output_mod,
                            squeeze_output=True)
        self.q2 = utils.MLP(dims, activation_fn=activation,
                            output_activation_fn=output_mod,
                            squeeze_output=True) if num_q == 2 else None
        self.infos = dict()
        self.apply(utils.weight_init)

    def forward(self, state, action):
        assert state.size()[:-1] == action.size()[:-1], print(state.size(), action.size())
        sa = torch.cat([state, action], 1)
        q1 = self.q1(sa) # q1: (batch_size,)
        self.infos['q1'] = q1
        if self.q2 is not None:
            q2 = self.q2(sa) # q2: (batch_size,)
            self.infos['q2'] = q2
            return q1, q2
        return q1

    def log(self, L, step, log_freq, param=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        if param:
            for i in range(self.hidden_depth+1):
                L.log_param('train_critic/q1_fc%d_param' % i,
                            self.q1[i * 2], step)
                L.log_param('train_critic/q2_fc%d_param' % i,
                            self.q2[i * 2], step) if self.q2 is not None else 0


class TwinCritic(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 l: int = 2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * l), 1]
        self.q1 = utils.MLP(dims, squeeze_output=True)
        self.q2 = utils.MLP(dims, squeeze_output=True)
        self.hidden_depth = l
        self.infos = dict()

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        q1, q2 = self.q1(sa), self.q2(sa)
        self.infos['q1'] = q1
        self.infos['q2'] = q2
        return q1, q2

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))

    def log(self, L, step, log_freq, param=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        if param:
            for i in range(self.hidden_depth+1):
                L.log_param('train_critic/q1_fc%d_param' % i, self.q1[i * 2], step)
                L.log_param('train_critic/q2_fc%d_param' % i, self.q2[i * 2], step)


class EnsembleCritic(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 1024,
                 l: int = 2, output_mod: Callable[[], nn.Module] = None,
                 num_q: int = 2, output_dim: int = 1, activation: str = 'relu'):
        super().__init__()
        self.output_dim = output_dim
        self.num_q = num_q
        self.hidden_depth = l

        dims = [state_dim + action_dim, *([hidden_dim] * l), output_dim]
        self.q = utils.MLP(dims, activation, output_mod, squeeze_output=True, ensemble_size=num_q)
        self.infos = dict()
        self.apply(utils.weight_init)

    def forward(self, state, action, minimize=True):
        assert state.size()[:-1] == action.size()[:-1], print(state.size(), action.size())
        state_action = torch.cat([state, action], -1)
        # [num_q, batch_size, state_dim + action_dim]
        state_action = state_action.unsqueeze(0).repeat_interleave(self.num_q, dim=0)
        # [num_q, batch_size]
        q = self.q(state_action)

        for i in range(q.size(0)):
            self.infos['q%s' % (i + 1)] = q[i]

        if minimize:
            # [batch_size, ]
            q = q.min(dim=0)[0] if q.size(0) == self.num_q else q
            self.infos['q_min'] = q
        return q

    def log(self, L, step, log_freq, param=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        if param:
            for i in range(self.hidden_depth+1):
                L.log_param('train_critic/ensemble_q_fc%d_param' % i, self.q[i * 2], step)


class VectorizedCritic(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 l: int = 2, num_q: int = 5, output_dim: int = 1):
        super().__init__()
        self.output_dim = output_dim
        self.num_q = num_q
        self.hidden_depth = l
        dims = [state_dim + action_dim, *([hidden_dim] * l), output_dim]
        self.q = utils.MLP(dims, activation_fn="relu", squeeze_output=True,
                           ensemble_size=num_q)

        # init as in the EDAC paper
        for layer in self.q[:-1][::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.q[-2].weight, -3.e-3, 3.e-3)
        torch.nn.init.uniform_(self.q[-2].bias, -3.e-3, 3.e-3)
        self.infos = dict()

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        # [num_q, batch_size, state_dim + action_dim]
        state_action = state_action.unsqueeze(0).repeat_interleave(self.num_q, dim=0)
        # [num_q, batch_size]
        q_values = self.q(state_action)

        for i in range(q_values.size(0)):
            self.infos['q%s' % (i + 1)] = q_values[i]

        return q_values

    def log(self, L, step, log_freq, param=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        if param:
            for i in range(self.hidden_depth+1):
                L.log_param('train_critic/ensemble_q_fc%d_param' % i, self.q[i * 2], step)


class ValueFunction(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int = 256, l: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * l), 1]
        self.v = utils.MLP(dims, squeeze_output=True)
        self.hidden_depth = l
        self.infos = dict()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        v = self.v(state)
        self.infos['v'] = v
        return v

    def log(self, L, step, log_freq, param=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_value/%s_hist' % k, v, step)

        if param:
            for i in range(self.hidden_depth+1):
                L.log_param('train_value/v_fc%d_param' % i, self.v[i * 2], step)


class EnsembleValue(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int = 1024, l: int = 2,
                 output_mod: Callable[[], nn.Module] = None, num_q: int = 2,
                 output_dim: int = 1, activation: str = 'relu'):
        super().__init__()
        self.output_dim = output_dim
        self.num_q = num_q
        self.hidden_depth = l

        dims = [state_dim, *([hidden_dim] * l), output_dim]
        self.v = utils.MLP(dims, activation_fn=activation, output_activation_fn=output_mod,
                           squeeze_output=True, ensemble_size=num_q)
        self.infos = dict()
        self.apply(utils.weight_init)

    def forward(self, state, mode=False):
        # [num_q, batch_size, state_dim]
        state = state.unsqueeze(0).repeat_interleave(self.num_q, dim=0)
        v = self.v(state) # (num_q, batch_size)

        if v.ndim > 1:
            for i in range(v.size(0)):
                self.infos['v%s' % (i + 1)] = v[i]

        if v.size(0) == 1 or mode == False:
            return v.squeeze(0)
        elif mode == 'random':
            idx = np.random.choice(v.size(0))
            v = v[idx]
        elif mode == 'min':
            v = v.min(0)[0]
        else:
            raise ValueError(mode)

        self.infos['v'] = v # (batch_size,)
        return v

    def log(self, L, step, log_freq, param=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_value/%s_hist' % k, v, step)

        if param:
            for i in range(self.hidden_depth+1):
                L.log_param('train_value/ensemble_v_fc%d_param' % i, self.v[i * 2], step)


class ObservationFunction(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 256, l: int = 2,
                 num_ensemble: int = 5, output_dim: int = 1, ensemble: bool = True):
        super().__init__()
        self.output_dim = output_dim
        self.num_ensemble = num_ensemble
        self.hidden_depth = l
        dims = [input_dim, *([hidden_dim] * l), output_dim]
        self.net = utils.MLP(dims, activation_fn="relu", squeeze_output=True,
                             ensemble_size=num_ensemble if ensemble else None)

        # init as in the EDAC paper
        for layer in self.q[:-1][::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.q[-2].weight, -3.e-3, 3.e-3)
        torch.nn.init.uniform_(self.q[-2].bias, -3.e-3, 3.e-3)
        self.infos = dict()

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        # [num_q, batch_size, state_dim + action_dim]
        state_action = state_action.unsqueeze(0).repeat_interleave(self.num_q, dim=0)
        # [num_q, batch_size]
        q_values = self.q(state_action)

        for i in range(q_values.size(0)):
            self.infos['q%s' % (i + 1)] = q_values[i]

        return q_values

    def log(self, L, step, log_freq, param=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        if param:
            for i in range(self.hidden_depth+1):
                L.log_param('train_critic/ensemble_q_fc%d_param' % i, self.q[i * 2], step)
