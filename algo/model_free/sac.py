import copy
import torch
import torch.nn.functional as F
import numpy as np

from algo.base_algo import BaseAlgorithm
from model.modules import SGMLPActor, Critic, EnsembleCritic
from common.utils import update_params, soft_update_params

_AVAILABLE_Q = {'ensemble': EnsembleCritic, 'normal': Critic}


class SAC(BaseAlgorithm):

    def __init__(self, state_dim, action_dim, action_limit, actor_lr=1e-3, alpha_lr=1e-3,
                 actor_update_freq=1, critic_lr=1e-3, critic_tau=0.005, backup_entropy=True,
                 critic_target_update_freq=1, num_q=2, gamma=0.99, l=2, hidden_dim=1024,
                 init_temperature=0.1, target_entropy=-3., behavior_cloning=False,
                 critic_type='ensemble', device='cpu', activation='relu', **kwargs):
        super().__init__(
            state_dim, action_dim, action_limit, actor_update_freq,
            critic_tau, critic_target_update_freq, device, gamma
        )
        # Setting modules
        self.actor = SGMLPActor(self.state_dim, self.action_dim, hidden_dim,
                                l, self.action_limit).to(device)
        self.critic = _AVAILABLE_Q[critic_type](self.state_dim, self.action_dim,
                                                hidden_dim, l, num_q=num_q,
                                                activation=activation).to(device)
        self.critic_targ = copy.deepcopy(self.critic)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = target_entropy if target_entropy is not None else -np.prod(action_dim)

        # Setting optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(0.9, 0.999))
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(0.9, 0.999))
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(0.9, 0.999))

        # Setting hyperparameters
        self.backup_entropy = backup_entropy
        self.behavior_cloning = behavior_cloning

        self.train()
        self.train_targ()
        self.print_module()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update_critic(self, s, a, r, s2, nd):
        self.update_critic_steps += 1

        q = self.critic(s, a, False)
        with torch.no_grad():
            _, a2, logp_a2, _ = self.actor(s2)
            q_pi_targ = self.critic_targ(s2, a2)
            if self.backup_entropy:
                q_pi_targ -= self.alpha * logp_a2
            q_targ = r + self.gamma * nd * q_pi_targ

        # loss_q = F.mse_loss(q, q_targ.unsqueeze(0).expand(*q.size()))
        loss_q = (q - q_targ.unsqueeze(0).detach()).pow(2).mean(-1)

        q_info_dict = dict(Qvals=q.min(0)[0], Qmaxs=q.max(0)[0],
                           TQvals=q_targ, LossQ=loss_q.mean().item())
        return loss_q.sum(), q_info_dict

    def update_actor(self, s, a):
        self.update_actor_steps += 1

        _, pi_a, logp_pi_a, _ = self.actor(s)

        # loss_alpha = (self.alpha * (-logp_pi_a - self.target_entropy).detach()).mean()
        loss_alpha = (self.log_alpha * (-logp_pi_a - self.target_entropy).detach()).mean()
        update_params(self.log_alpha_optimizer, loss_alpha)

        q_pi = self.critic(s, pi_a)
        loss_pi = (self.alpha.detach() * logp_pi_a - q_pi).mean()
        if self.behavior_cloning and self.behavior_cloning > self.total_time_steps:
            logp_a = self.actor.log_prob(s, a)
            loss_pi = (self.alpha.detach() * logp_pi_a - logp_a).mean()

        pi_info_dict = dict(
            LossPi=loss_pi.item(), Alpha=self.alpha.item(), HPi=-logp_pi_a, LossAlpha=loss_alpha.item())
        return loss_pi, pi_info_dict


    def _update(self, data, logger, step, save_log=False):
        self.update_steps += 1

        s, a, r, s2, nd = data['state'], data['action'], data['reward'], data['next_state'], data['not_done']
        r, nd = r.view(-1), nd.view(-1)

        '''Update critic'''
        loss_q, q_info_dict = self.update_critic(s, a, r, s2, nd)
        update_params(self.critic_optimizer, loss_q)
        if save_log:
            self.critic.log(logger['tb'], step, True)

        '''Update actor'''
        if step % self.actor_update_freq == 0:
            loss_pi, pi_info_dict = self.update_actor(s, a)
            update_params(self.actor_optimizer, loss_pi)
            if save_log:
                self.actor.log(logger['tb'], step, True)

        '''Smooth update'''
        if step % self.critic_target_update_freq == 0:
            soft_update_params(self.critic, self.critic_targ, self.critic_tau)

        return q_info_dict, pi_info_dict

    def _save(self, model_dir, step):
        pass

    def _load(self, model_dir, step):
        pass

    def _print_log(self, logger):
        logger.log_tabular('Qmaxs', average_only=True)
        logger.log_tabular('HPi', average_only=True)
        logger.log_tabular('Alpha', average_only=True)
        logger.log_tabular('LossAlpha', average_only=True)
