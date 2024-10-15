import re
import os
import gym
import yaml
import uuid
import wandb
import math
import time
import datetime
import random
import numpy as np
from pathlib import Path
import torch
from torch import nn
from torch import distributions
import torchvision
import torch.nn.functional as F

from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def asdict(config):
    dic = {}
    # config_dict = config.__dict__
    for key, value in config.items():
        if not key.startswith('__'):
            dic[key] = value
    return dic


def wandb_init(config: dict) -> None:
    config['corruption_rate'] = config['corrupt_cfg']['corruption_rate']
    config['corruption_mode'] = config["corrupt_cfg"]["corruption_mode"],
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["algo_name"],
        id=str(uuid.uuid4()),
    )


def wrap_env(env: gym.Env,
             state_mean: Union[np.ndarray, float] = 0.0,
             state_std: Union[np.ndarray, float] = 1.0,
             reward_scale: float = 1.0) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def log(logger, key, value_dict, epoch):
    assert isinstance(logger, dict) and isinstance(value_dict, dict), \
            print(type(logger), type(value_dict))
    logger['tb'].save_log_dict(key, value_dict, epoch)
    logger['sp'].store(**value_dict)


def figure_to_numpy(fig):
    # fig = plt.gcf()
    bbox = fig.get_window_extent()
    width, height = bbox.width, bbox.height
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(int(height), int(width), 3)
    np_fig = np.array(data)
    return np_fig


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize(value: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (value - mean) / std


def np2torch(x, device):
    return torch.as_tensor(x).to(device)


def log_sum_exp(value, dim):
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value_norm = value - m
    m = m.squeeze(dim)
    return m+torch.log(torch.sum(torch.exp(value_norm), dim=dim))


def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5 * torch.log(one_plus_x/ one_minus_x)


def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
        log_pi = log_pi.squeeze(-1)
    return mu, pi, log_pi


# https://github.com/xtma/dsac/blob/master/rlkit/torch/dsac/dsac.py
def calc_presum_tau(batch_size, num_quantiles, device, algo):
    assert batch_size != 0 and num_quantiles != 0 and device != None and algo != 'fqf'
    if 'qr' in algo:
        presum_tau = torch.zeros(batch_size, num_quantiles, device=device) + 1. / num_quantiles
    elif 'iqn' in algo:
        presum_tau = torch.rand(batch_size, num_quantiles, device=device) + 0.1
        presum_tau /= presum_tau.sum(dim=-1, keepdims=True)
    else:
        raise NotImplementedError('get_tau must implemented under qr or iqn without presum_tau')
    return presum_tau


def get_tau(batch_size=0, num_quantiles=0, device=None, algo='qr', presum_tau=None):
    if presum_tau is None:
        presum_tau = calc_presum_tau(batch_size, num_quantiles, device, algo)
    tau = torch.cumsum(presum_tau, dim=1)
    with torch.no_grad():
        tau_hat = torch.zeros_like(tau)
        tau_hat[:, 0:1] = tau[:, 0:1] / 2.
        tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
    return tau, tau_hat, presum_tau


def qr_loss(inputs, targets, tau, weight, sigma=1, reduction=True):
    # [num_q, batch_size, num_quantiles, 1]
    inputs = inputs.unsqueeze(-1)
    # [batch_size, 1, num_quantiles]
    targets = targets.unsqueeze(-2)
    # [num_q, batch_size, num_quantiles, 1]
    tau = tau.detach().unsqueeze(-1)
    # [num_q, batch_size, 1, num_quantiles]
    # weight = weight.detach().unsqueeze(-2)
    # if weight.ndim == 3:
    #     # [1, batch_size, 1, num_quantiles]
    #     weight = weight.unsqueeze(0)
    # [num_q, batch_size, num_quantiles, num_quantiles]
    expanded_inputs, expanded_targets = torch.broadcast_tensors(inputs, targets)

    # L = F.smooth_l1_loss(expanded_inputs, expanded_targets, reduction="none")
    L = huber_loss(expanded_inputs - expanded_targets, sigma=sigma)

    sign = torch.sign(expanded_inputs - expanded_targets) / 2. + 0.5
    rho = torch.abs(tau - sign) * L # * weight
    if reduction == "none":
        return rho.sum(dim=-1)
    return rho.sum(dim=-1).mean()


def huber_loss(diff, sigma=1):
    beta = 1. / (sigma ** 2)
    diff = torch.abs(diff)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss


def normalize_torch_data(data: torch.Tensor, lower=0.0, upper=1.0):
    data_min, data_max = data.min(), data.max()
    norm_data = (data - data_min) / (data_max - data_min + 1e-6) * (upper - lower) + lower
    return norm_data


def update_params(optim, loss, retain_graph=False, grad_cliping=False, networks=None):

    if isinstance(optim, dict):
        for k, opt in optim.items():
            opt.zero_grad()
    else:
        optim.zero_grad()

    loss.backward(retain_graph=retain_graph)
    # Clip norms of gradients to stebilize training.
    if grad_cliping:
        try:
            for net in networks:
                nn.utils.clip_grad_norm_(net.parameters(), grad_cliping)
        except:
            nn.utils.clip_grad_norm_(networks.parameters(), grad_cliping)

    if isinstance(optim, dict):
        for k, opt in optim.items():
            opt.step()
    else:
        optim.step()


# https://github.com/facebookresearch/drqv2/blob/main/utils.py
def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


def linear_decay_fn(start_steps: int, max_steps: int,
                    initial_value: float, final_value: float):
    dur_step = max_steps - start_steps

    def linear_decay(current_step: int):
        if current_step <= start_steps:
            return initial_value

        if current_step >= max_steps:
            return final_value

        slope = (final_value - initial_value) / dur_step
        decayed_value = initial_value + slope * (current_step - start_steps)
        return decayed_value
    return linear_decay


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1.0 - tau) * target_param.data
        )


def calc_time(start_time):
    return str(datetime.timedelta(seconds=int(time.time() - start_time)))


def yaml_write_dict(write_dir, name, dictionary):
    write_dir = Path(write_dir)
    assert write_dir.is_dir(), print(write_dir)
    fp = open(write_dir / name, 'w')
    fp.write(yaml.dump(dictionary))
    fp.close()


def yaml_load_dict(load_dir, name):
    load_dir = Path(load_dir)
    assert load_dir.is_dir(), print(load_dir)
    with open(load_dir / name) as f:
        dictionary = yaml.unsafe_load(f)
    return dictionary


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            if model is not None:
                self.prev_states.append(model.training)
                model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    receptive_field_size = 1
    if dimensions == 2:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
    elif dimensions >= 3:
        num_input_fmaps = tensor.size(2)
        num_output_fmaps = tensor.size(1)
        if tensor.dim() > 3:
            # math.prod is not always available, accumulate the product manually
            # we could use functools.reduce but that is not supported by TorchScript
            for s in tensor.shape[3:]:
                receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.) -> torch.Tensor:
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    with torch.no_grad():
        return tensor.uniform_(-a, a)

def weight_init(m):
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    if isinstance(m, EnsembleLinear) or isinstance(m, nn.Linear):
        xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

def weight_init_v2(m):
    if isinstance(m, EnsembleLinear) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        nn.init.constant_(m.bias, 0)
        # if m.bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #     nn.init.uniform_(m.bias, -bound, bound)

def weight_init_v3(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if isinstance(m, EnsembleLinear) or isinstance(m, nn.Linear):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def fanin_init(m):
    if isinstance(m, EnsembleLinear) or isinstance(m, nn.Linear):
        size = m.weight.size()
        if len(size) == 2:
            fan_in = size[0]
        elif len(size) > 2:
            fan_in = np.prod(size[1:])
        else:
            raise Exception("Shape must be have dimension at least 2.")
        bound = 1. / np.sqrt(fan_in)
        m.weight.data.uniform_(-bound, bound)
        if m.bias is not None:
            m.bias.data.fill_(0.1)

def edac_init(m):
    # init as in the EDAC paper
    if isinstance(m, EnsembleLinear) or isinstance(m, nn.Linear):
        torch.nn.init.constant_(m.bias, 0.1)


class EnsembleLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, ensemble_size: int,
                 weight_decay: float = 0., bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight_decay = weight_decay

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            # input: [ensemble_size, batch_size, input_size]
            # weight: [ensemble_size, input_size, out_size]
            # out: [ensemble_size, batch_size, out_size]
            return x @ self.weight + self.bias if self.bias is not None \
                    else x @ self.weight
        elif x.ndim == 4:
            # input: [ensemble_size, batch_size, hidden_size, input_size]
            # weight: [ensemble_size, 1, input_size, out_size]
            # out: [ensemble_size, batch_size, hidden_size, out_size]
            weight = self.weight.unsqueeze(1)
            return x @ weight + self.bias.unsqueeze(1) if self.bias is not None \
                    else x @ weight
        else:
            raise NotImplementedError(x.ndim)


    def get_params(self, indexes):
        assert len(indexes) <= self.ensemble_size and max(indexes) < self.ensemble_size
        if self.bias is not None:
            return self.weight.data[indexes], self.bias.data[indexes]
        return self.weight.data[indexes]

    def save_params(self, indexes, params):
        assert len(indexes) <= self.ensemble_size and max(indexes) < self.ensemble_size
        if self.bias is not None:
            weight, bias = params
            self.weight.data[indexes] = weight
            self.bias.data[indexes] = bias
        else:
            self.weight.data[indexes] = params

    def extra_repr(self):
        return 'in_features={}, out_features={}, ensemble_size={}, bias={}'.format(
            self.in_features, self.out_features, self.ensemble_size, self.bias is not None
        )


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class Swish(nn.Module):

    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(torch.sigmoid(x)) if self.inplace else x * torch.sigmoid(x)


_AVAILABLE_ACTIVATION = {
    'swish': Swish,
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
}


def MLP(dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        ensemble_size: Optional[int] = None):

    def linear(n_input, n_output):
        if ensemble_size is not None:
            return EnsembleLinear(n_input, n_output, ensemble_size)
        return nn.Linear(n_input, n_output)

    n_dims = len(dims)
    if n_dims < 2:
        raise ValueError("MLP requires at least two dims (input and output)")

    layers = []
    if isinstance(activation_fn, str):
        activation_fn = _AVAILABLE_ACTIVATION[activation_fn]
    for i in range(n_dims - 2):
        layers.append(linear(dims[i], dims[i + 1]))
        layers.append(activation_fn())
    layers.append(linear(dims[-2], dims[-1]))
    if output_activation_fn is not None:
        if isinstance(output_activation_fn, str):
            output_activation_fn = _AVAILABLE_ACTIVATION[output_activation_fn]
        layers.append(output_activation_fn())
    if squeeze_output:
        if dims[-1] != 1:
            raise ValueError("Last dim must be 1 when squeezing")
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    return net



import csv

CSV_HEAD = ['return', 'score', 'seed', 'corruption_mode', 'corruption_data', 'corruption_rate',
            'corruption_range','nstep', 'discount', 'algo', 'batch_size', 'epoch', 'step', 'total_time']

def save_log_in_csv(eval_info: dict, write_dir: str, max_episode: int, cfg: dict):
    write_dir = f"{write_dir}-{max_episode}.csv"
    try:
        with open(write_dir, 'r') as f:
            line = f.readline()
            write_head = line[:6] != CSV_HEAD[0]
    except:
        write_head = True

    with open(write_dir, 'a') as f:
        csv_writer = csv.DictWriter(f, fieldnames=CSV_HEAD)
        data = {}
        if write_head:
            for key in CSV_HEAD:
                data[key] = key
            csv_writer.writerow(data)
            data = {}
        data['return'] = eval_info['TestEpRet']
        data['score'] = eval_info['TestScore']
        data['seed'] = cfg.seed
        data['corruption_mode'] = cfg.corrupt_cfg.corruption_mode
        data['corruption_data'] = cfg.corrupt_cfg.corruption_type
        data['corruption_rate'] = cfg.corrupt_cfg.corruption_rate
        data['corruption_range'] = cfg.corrupt_cfg.corruption_range
        data['nstep'] = 1 #cfg.nstep
        data['discount'] = cfg.gamma
        data['algo'] = cfg.algo_name
        data['batch_size'] = cfg.batch_size
        data['epoch'] = eval_info['epoch']
        data['step'] = eval_info['step']
        data['total_time'] = eval_info['total_time']
        csv_writer.writerow(data)
    print(max_episode, "Return: ", eval_info['TestEpRet'], " Score: ", eval_info['TestScore'])
