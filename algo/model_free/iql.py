import copy
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from algo.base_algo import BaseAlgorithm
from model.modules import GaussianActor, TwinCritic, ValueFunction
from common.utils import update_params, soft_update_params


class IQL(BaseAlgorithm):

    def __init__(self, state_dim, action_dim, action_limit, actor_lr=1e-3,
                 actor_update_freq=1, critic_lr=1e-3, critic_tau=0.005,
                 critic_target_update_freq=1, gamma=0.99, l=2, hidden_dim=1024,
                 device='cpu', beta=3., iql_tau=0.7, clip_score=100, max_epochs=1000,
                 step_per_epoch=1000, **kwargs):
        super().__init__(
            state_dim, action_dim, action_limit, actor_update_freq,
            critic_tau, critic_target_update_freq, device, gamma
        )
        # Setting modules
        self.actor = GaussianActor(self.state_dim, self.action_dim, hidden_dim,
                                    l, self.action_limit).to(device)
        self.critic = TwinCritic(self.state_dim, self.action_dim, hidden_dim, l).to(device)
        self.critic_targ = copy.deepcopy(self.critic).requires_grad_(False).to(device)
        self.value = ValueFunction(self.state_dim, hidden_dim, l).to(device)

        # Setting optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=critic_lr)
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_epochs*step_per_epoch)

        # Setting hyperparameters
        self.beta = beta
        self.iql_tau = iql_tau
        self.clip_score = clip_score

        self.train()
        self.train_targ()
        self.print_module()


    def update_actor(self, s, a, adv):
        self.update_actor_steps += 1
        loss_pi, pi_info_dict = self.advantage_weighted_regression(s, a, adv)
        update_params(self.actor_optimizer, loss_pi)
        self.actor_lr_schedule.step()
        return pi_info_dict

    def advantage_weighted_regression(self, s, a, adv):
        policy_out = self.actor(s)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(a)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != a.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - a) ** 2, dim=1)
        else:
            raise NotImplementedError
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=self.clip_score)
        loss_pi = torch.mean(exp_adv * bc_losses)

        pi_info_dict = dict(LossPi=loss_pi.item(), Weight=exp_adv.detach(), HPi=bc_losses)
        return loss_pi, pi_info_dict

    def update_critic(self, s, a, r, s2, nd):
        self.update_critic_steps += 1

        loss_v, v_info_dict = self.update_vfunc(s, a)
        update_params(self.value_optimizer, loss_v)

        loss_q, q_info_dict = self.update_qfunc(s, a, r, s2, nd)
        update_params(self.critic_optimizer, loss_q)
        return q_info_dict, v_info_dict

    def update_vfunc(self, s, a):
        target_q = self.critic_targ(s, a).detach()
        v = self.value(s) # (batch_size,)
        adv = target_q - v
        loss_v = torch.mean(torch.abs(self.iql_tau - (adv < 0).float()) * adv**2)

        v_info_dict = dict(Vals=v, TVals=target_q, LossV=loss_v.item(), Adv=adv)
        return loss_v, v_info_dict

    def update_qfunc(self, s, a, r, s2, nd):
        q1, q2 = self.critic.both(s, a) # (num_q, batch_size)
        v_pi_targ = self.value(s2).detach() # (batch_size,)
        q_targ = r + self.gamma * nd * v_pi_targ # (batch_size,)

        loss_q = 0.5 * (F.mse_loss(q1, q_targ.detach()) + F.mse_loss(q2, q_targ.detach()))
        q_info_dict = dict(Qvals=torch.min(q1, q2), Qmaxs=torch.max(q1, q2),
                           TQvals=q_targ, LossQ=loss_q.item())
        return loss_q, q_info_dict


    def _update(self, data, logger, step, save_log=False):
        self.update_steps += 1

        s, a, r, s2, nd = data['state'], data['action'], data['reward'], data['next_state'], data['not_done']
        r, nd = r.squeeze(-1), nd.squeeze(-1)
        '''Calc loss of qfunc and vfunc'''
        q_info_dict, v_info_dict = self.update_critic(s, a, r, s2, nd)

        '''Calc loss of actor'''
        if step % self.actor_update_freq == 0:
            adv = v_info_dict['Adv'].detach()
            pi_info_dict = self.update_actor(s, a, adv)

        '''Save log_info'''
        if save_log:
            self.critic.log(logger['tb'], step, True)
            self.value.log(logger['tb'], step, True)
            self.actor.log(logger['tb'], step, True)

        '''Smooth update'''
        if step % self.critic_target_update_freq == 0:
            soft_update_params(self.critic, self.critic_targ, self.critic_tau)

        q_info_dict.update(v_info_dict)
        return q_info_dict, pi_info_dict


    def _save(self, model_dir, step):
        pass

    def _load(self, model_dir, step):
        pass

    def _print_log(self, logger):
        logger.log_tabular('Qmaxs', average_only=True)
        logger.log_tabular('Vals', average_only=True)
        logger.log_tabular('TVals', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('Adv', average_only=True)
        logger.log_tabular('Weight', average_only=True)
        logger.log_tabular('HPi', average_only=True)
