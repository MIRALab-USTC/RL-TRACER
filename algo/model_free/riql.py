import copy
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from algo.base_algo import BaseAlgorithm
from model.modules import TanhGaussianActor, GaussianActor, VectorizedCritic, ValueFunction
from common.utils import huber_loss, update_params, soft_update_params


class RIQL(BaseAlgorithm):

    def __init__(self, state_dim, action_dim, action_limit, actor_lr=1e-3,
                 actor_update_freq=1, critic_lr=1e-3, critic_tau=0.005,
                 critic_target_update_freq=1, num_q=2, gamma=0.99, l=2,
                 hidden_dim=256, device='cpu', beta=3., quantile=0.25,
                 iql_tau=0.7, clip_score=100, sigma=1.0, max_epochs=1000,
                 step_per_epoch=1000, **kwargs):
        super().__init__(
            state_dim, action_dim, action_limit, actor_update_freq,
            critic_tau, critic_target_update_freq, device, gamma
        )
        # Setting modules
        self.actor = GaussianActor(self.state_dim, self.action_dim, hidden_dim,
                                   l, self.action_limit).to(device)
        self.critic = VectorizedCritic(self.state_dim, self.action_dim,
                                       hidden_dim, l, num_q=num_q).to(device)
        self.critic_targ = copy.deepcopy(self.critic).requires_grad_(False).to(device)
        self.value = ValueFunction(self.state_dim, hidden_dim, l).to(device)

        # Setting optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=critic_lr)
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_epochs*step_per_epoch)

        # Setting hyperparameters
        self.beta = beta
        self.quantile = quantile
        self.iql_tau = iql_tau
        self.clip_score = clip_score
        self.sigma = sigma
        self.attack_infor = False

        self.train()
        self.train_targ()
        self.print_module()

    def _print_module(self):
        print("Quantile: ", self.quantile)
        print("IQL-Tau: ", self.iql_tau)
        print("Sigma: ", self.sigma)
        print("Beta: ", self.beta)

    def update_actor(self, s, a, adv):
        self.update_actor_steps += 1
        loss_pi, pi_info_dict = self.advantage_weighted_regression(s, a, adv)
        update_params(self.actor_optimizer, loss_pi)
        self.actor_lr_schedule.step()
        return pi_info_dict

    def advantage_weighted_regression(self, s, a, adv):
        if isinstance(self.actor, GaussianActor):
            loss_bc = -self.actor(s).log_prob(a)
        elif isinstance(self.actor, TanhGaussianActor):
            loss_bc = -self.actor.log_prob(s, a)
        else:
            raise NotImplementedError
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=self.clip_score)
        loss_pi = torch.mean(exp_adv * loss_bc)

        pi_info_dict = dict(LossPi=loss_pi.item(), Weight=exp_adv.detach(),
                            HPi=loss_bc, LossBC=loss_bc.mean().item())
        return loss_pi, pi_info_dict

    def update_critic(self, s, a, r, s2, nd, attack_indexes, step):
        self.update_critic_steps += 1

        loss_v, v_info_dict = self.update_vfunc(s, a, attack_indexes)
        update_params(self.value_optimizer, loss_v)

        loss_q, q_info_dict = self.update_qfunc(s, a, r, s2, nd, attack_indexes, step)
        update_params(self.critic_optimizer, loss_q)
        return q_info_dict, v_info_dict

    def update_vfunc(self, s, a, attack_indexes) -> tuple:
        v_info_dict = dict()
        # Update value function
        with torch.no_grad():
            ### calculate target Q with quantile estimator
            target_q_all = self.critic_targ(s, a)
            target_q = torch.quantile(target_q_all, self.quantile, dim=0)

            target_q_std = target_q_all.std(dim=0)
            target_diff = target_q_all.mean(dim=0) - target_q

        self.attack_infor = (attack_indexes is not None)
        if self.attack_infor:
            v_info_dict.update(dict(
                attack_TQvals_std = target_q_std[torch.where(attack_indexes == 1)],
                clean_TQvals_std = target_q_std[torch.where(attack_indexes == 0)],
                attack_TQvals_diff = target_diff[torch.where(attack_indexes == 1)],
                clean_TQvals_diff = target_diff[torch.where(attack_indexes == 0)]
            ))

        v = self.value(s) # (batch_size,)
        adv = target_q - v
        loss_v = torch.mean(torch.abs(self.iql_tau - (adv < 0).float()) * adv**2)

        v_info_dict.update(dict(Vals=v, TVals=target_q, LossV=loss_v.item(), Adv=adv))
        return loss_v, v_info_dict

    def update_qfunc(self, s, a, r, s2, nd, attack_indexes, step):
        with torch.no_grad():
            v_pi_targ = self.value(s2) # (batch_size,)
            q_targ = r + nd * self.gamma * v_pi_targ
            # target clipping
            q_targ = torch.clamp(q_targ, -100, 1000).view(1, q_targ.shape[0])

        q_vals = self.critic(s, a) # (num_q, batch_size)
        # Huber loss for Q functions
        assert q_vals.ndim == q_targ.ndim and q_vals.size(-1) == q_targ.size(-1), \
                print(q_vals.shape, q_targ.shape)
        loss_q = huber_loss(q_targ - q_vals, sigma=self.sigma).mean()

        q_info_dict = dict(LossQ=loss_q.item(), Qvals=q_vals.mean(0),
                           TQvals=q_targ.mean(0))
        return loss_q, q_info_dict

    def _update(self, data, logger, step, save_log=False):
        self.update_steps += 1

        s, a, s2 = data['state'], data['action'], data['next_state']
        r, nd = data['reward'].squeeze(-1), data['not_done'].squeeze(-1)
        attack_indexes = data['attack_indexes'] if 'attack_indexes' in data else None
        '''Calc loss of qfunc and vfunc'''
        q_info_dict, v_info_dict = self.update_critic(s, a, r, s2, nd, attack_indexes, step)

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
        logger.log_tabular('Vals', average_only=True)
        logger.log_tabular('TVals', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('Adv', average_only=True)
        logger.log_tabular('Weight', average_only=True)
        logger.log_tabular('HPi', average_only=True)
        logger.log_tabular('LossBC', average_only=True)
        if self.attack_infor:
            logger.log_tabular('attack_TQvals_diff', average_only=True)
            logger.log_tabular('clean_TQvals_diff', average_only=True)
            logger.log_tabular('attack_TQvals_std', average_only=True)
            logger.log_tabular('clean_TQvals_std', average_only=True)
