import copy
import torch
import torch.nn.functional as F

from algo.base_algo import BaseAlgorithm
from model.modules import MLPActor, Critic
from common.utils import update_params, soft_update_params


class TD3(BaseAlgorithm):

    def __init__(self, observation_space, action_space, actor_lr=1e-3,
                 actor_update_freq=1, act_noise=0.1, critic_lr=1e-3,
                 critic_tau=0.005, critic_target_update_freq=1,
                 device='cpu', gamma=0.99, hidden_dim=1024, l=2,
                 noise_clip=0.5, targ_noise=0.2, **kwargs):
        super().__init__(
            observation_space, action_space, actor_update_freq,
            critic_tau, critic_target_update_freq, device, gamma
        )
        # Setting hyperparameters
        self.noise_clip = noise_clip
        self.targ_noise = targ_noise
        
        # Setting modules
        self.actor = MLPActor(self.state_dim, self.action_dim, hidden_dim,
                              l, self.action_limit, act_noise).to(device)
        self.critic = Critic(
            self.state_dim, self.action_dim, hidden_dim, l).to(device)
        self.actor_targ = copy.deepcopy(self.actor)
        self.critic_targ = copy.deepcopy(self.critic)

        # Setting optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(0.9, 0.999))
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(0.9, 0.999))

        self._train()
        self._train_targ()
        self.print_module()

    def update_critic(self, o, a, r, o2, nd):
        self.update_critic_steps += 1

        q1, q2 = self.critic(o, a)
        q = torch.min(q1, q2).detach()

        with torch.no_grad():
            a2 = self.actor_targ(
                o2, act_noise=self.targ_noise, clip=True, noise_clip=self.noise_clip
            )
            q1_pi_targ, q2_pi_targ = self.critic_targ(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            q_targ = r + self.gamma * nd * q_pi_targ

        loss_q1, loss_q2 = F.mse_loss(q1, q_targ), F.mse_loss(q2, q_targ)
        loss_q = loss_q1 + loss_q2
        return loss_q, dict(Q1=q1, Q2=q2, Qvals=q, TQvals=q_targ,
                            LossQ=loss_q.item(), LossQ1=loss_q1.item(), LossQ2=loss_q2.item())

    def update_actor(self, o):
        self.update_actor_steps += 1
        loss_pi = -self.critic.Q1(o, self.actor(o, True)).mean()
        return loss_pi, dict(LossPi=loss_pi.item())

    def _update(self, data, logger, step, save_log=False):
        self.update_steps += 1
        o, a, r, o2, nd = data['state'], data['action'], data['reward'], data['next_state'], data['not_done']
        r, nd = r.view(-1), nd.view(-1)
        # Update Q
        loss_q, q_info_dict = self.update_critic(o, a, r, o2, nd, logger)
        update_params(self.critic_optimizer, loss_q)
        if save_log:
            self.critic.log(logger['tb'], step, True)
        # Update Pi
        if step % self.actor_update_freq == 0:
            loss_pi, pi_info_dict = self.update_actor(s)
            update_params(self.actor_optimizer, loss_pi)
            if save_log:
                self.actor.log(logger['tb'], step, True)
        # Update the frozen target models
        if step % self.critic_target_update_freq == 0:
            soft_update_params(self.critic, self.critic_targ, self.critic_tau)
            soft_update_params(self.actor, self.actor_targ, self.critic_tau)
        return q_info_dict, pi_info_dict

    def _save(self, filename):
        pass

    def _load(self, filename):
        pass

    def _print_log(self, logger):
        logger.log_tabular('Q1', average_only=True)
        logger.log_tabular('Q2', average_only=True)
        logger.log_tabular('LossQ1', average_only=True)
        logger.log_tabular('LossQ2', average_only=True)
