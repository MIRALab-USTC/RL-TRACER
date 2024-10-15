import copy
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import wandb
import matplotlib.pyplot as plt

from common import utils
from algo.base_algo import BaseAlgorithm
from model.modules import TanhGaussianActor, GaussianActor
from model.bayes_modules import DistributionalCritic, DistributionalValueFunction, ObservationModel
from common.utils import qr_loss, update_params


class TRACER(BaseAlgorithm):

    def __init__(
            self,
            state_dim,
            action_dim,
            action_limit,
            actor_lr=1e-3,
            actor_update_freq=1,
            critic_lr=1e-3,
            critic_tau=0.005,
            critic_target_update_freq=1,
            num_q=2,
            q_std=[-5, 2],
            cosines_dim=64,
            num_quantiles=32,
            gamma=0.99,
            l=2,
            hidden_dim=256,
            critic_init_type="edac",
            device='cpu',
            beta=3.,
            quantile=0.25,
            iql_tau=0.7,
            clip_score=100,
            sigma=1.0,
            max_epochs=3000,
            step_per_epoch=1000,
            num_model=1,
            model_lr=1e-3,
            obser_sigma=0.3,
            obser_model_type="gaussian",
            obser_beta=[0.0001, 0.01],
            obser_beta_epoch=[5, 2000],
            enable_entropy=False,
            lower_bound=0.9,
            upper_bound=1.0,
            load_obser_model=False,
            obser_model_dir=None,
            obser_model_step=-1,
            activation="relu",
            std_architecture="mlp",
            **kwargs):
        super().__init__(
            state_dim, action_dim, action_limit, actor_update_freq,
            critic_tau, critic_target_update_freq, device, gamma
        )
        # Setting modules
        self.actor = GaussianActor(self.state_dim, self.action_dim, hidden_dim,
                                   l, self.action_limit).to(device)
        self.critic = DistributionalCritic(self.state_dim, self.action_dim + 1,
                                           hidden_dim, l, num_q, cosines_dim,
                                           init_type=critic_init_type).to(device)
        self.critic_targ = copy.deepcopy(self.critic).requires_grad_(False).to(device)
        self.value = DistributionalValueFunction(self.state_dim, hidden_dim, l, cosines_dim,
                                                 init_type=critic_init_type).to(device)

        # Initial model for observable variables
        self.obser_model = ObservationModel(self.state_dim, self.action_dim,
                                            1, num_quantiles, hidden_dim, l,
                                            num_model, obser_sigma, obser_model_type,
                                            activation, std_architecture).to(device)

        # Setting optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=critic_lr)
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_epochs*step_per_epoch)
        self.obser_optimizer = torch.optim.Adam(self.obser_model.parameters(), lr=model_lr)
        self.obser_beta_decay = utils.linear_decay_fn(obser_beta_epoch[0] * step_per_epoch,
                                                      obser_beta_epoch[1] * step_per_epoch,
                                                      obser_beta[0], obser_beta[1])

        # Setting hyperparameters
        self.beta = beta
        self.quantile = quantile
        self.iql_tau = iql_tau
        self.clip_score = clip_score
        self.sigma = sigma
        self.attack_infor = False
        self.num_q = num_q
        self.num_quantiles = num_quantiles
        self.enable_entropy = enable_entropy
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.train()
        self.train_targ()
        self.print_module()

    def _print_module(self):
        print("ObserModel: ", self.obser_model)
        print("Quantile: ", self.quantile)
        print("IQL-Tau: ", self.iql_tau)
        print("Sigma: ", self.sigma)
        print("Beta: ", self.beta)
        print("Num Quantiles: ", self.num_quantiles)

    def sample_tau(self, batch_size):
        _, tau_hat, presum_tau = utils.get_tau(
                batch_size, self.num_quantiles, self.device, 'iqn')
        return tau_hat, presum_tau

    def d_entropy(self, d_vals: torch.Tensor, presum_tau: torch.Tensor):
        size1, size2 = d_vals.size(0), d_vals.size(1)
        # Flatten the tensors for batch processing
        d = d_vals.view(-1, self.num_quantiles)
        pdf = presum_tau.view(-1, self.num_quantiles)

        # Sort the distributions and corresponding probabilities
        sorted_d, indices = torch.sort(d, dim=1)
        sorted_pdf = torch.gather(pdf, 1, indices)

        # Calculate the midpoint probabilities and the differences between sorted values
        sorted_d_diff = sorted_d[:, 1:] - sorted_d[:, :-1]
        mid_pdf = (sorted_pdf[:, 1:] + sorted_pdf[:, :-1]) / 2

        # Compute entropy efficiently
        log_mid_pdf = torch.log(mid_pdf)
        entropy = -torch.sum(sorted_d_diff * mid_pdf * log_mid_pdf, dim=1)

        # Reshape and normalize entropy
        entropy = entropy.view(size1, size2, 1)
        weight = torch.exp(- entropy / (sorted_d.mean() + 1e-6))
        norm_weight = utils.normalize_torch_data(weight, self.lower_bound, self.upper_bound)
        return entropy, norm_weight

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
        loss_pi = torch.mean(exp_adv * loss_bc.unsqueeze(-1))

        pi_info_dict = dict(LossPi=loss_pi.item(), Weight=exp_adv.detach().mean(-1),
                            HPi=loss_bc, LossBC=loss_bc.mean().item())
        return loss_pi, pi_info_dict

    def update_critic(self, s, a, r, s2, nd, attack_indexes, step):
        self.update_critic_steps += 1
        loss_v, v_info_dict = self.update_vfunc(s, a, r, attack_indexes)
        update_params(self.value_optimizer, loss_v)

        loss_q, q_info_dict = self.update_qfunc(s, a, r, s2, nd, attack_indexes, step)
        update_params(self.critic_optimizer, loss_q)
        return q_info_dict, v_info_dict

    def update_vfunc(self, s, a, r, attack_indexes) -> tuple:
        v_info_dict = dict()
        batch_size = s.size(0)
        num_quantiles = self.num_quantiles
        # Update value function
        with torch.no_grad():
            ### calculate target Q with quantile estimator
            # [num_q, batch_size, num_quantiles]
            target_tau_hat = self.sample_tau(batch_size * self.num_q)[0].view(
                    self.num_q, batch_size, num_quantiles)
            # [num_q, batch_size, num_quantiles]
            target_d_all = self.critic_targ(torch.cat([s, a, r], dim=-1), target_tau_hat)
            # [batch_size, num_quantiles]
            target_d = torch.quantile(target_d_all, self.quantile, dim=0)

        tau_hat, _ = self.sample_tau(batch_size)
        # [batch_size, num_quantiles]
        z_vals = self.value(s, tau_hat)
        adv = target_d - z_vals
        loss_v = torch.mean((self.iql_tau - (adv < 0).float()).abs() * adv.square())

        v_info_dict.update(dict(Vals=z_vals.mean(-1), TVals=target_d.mean(-1),
                                LossV=loss_v.item(), Adv=adv))
        return loss_v, v_info_dict

    def update_qfunc(self, s, a, r, s2, nd, attack_indexes, step):
        with torch.no_grad():
            # [batch_size, num_quantiles]
            tau_hat2, presum_tau2 = self.sample_tau(s2.size(0))
            # [batch_size, num_quantiles]
            z_pi_targ = self.value(s2, tau_hat2)
            d_targ = r + nd * self.gamma * z_pi_targ
            # target clipping
            d_targ = torch.clamp(d_targ, -100, 1000)

        # [num_q, batch_size, num_quantiles]
        tau_hat = self.sample_tau(s.size(0) * self.num_q)[0].view(
                self.num_q, s.size(0), self.num_quantiles)
        # [num_q, batch_size, num_quantiles]
        d_vals = self.critic(torch.cat([s, a, r], dim=-1), tau_hat)

        # Huber loss for Q functions
        loss_q = qr_loss(d_vals, d_targ, tau_hat, presum_tau2, self.sigma)

        q_info_dict = dict(LossQ=loss_q.item(), Qvals=d_vals.mean([-1, 0]),
                           TQvals=d_targ.mean(-1))
        return loss_q, q_info_dict

    def update_obser_model(self, s, a, r, s2, attack_indexes, step):
        '''Reconstruction Data'''
        with torch.no_grad():
            data = torch.cat([s, a, r], dim=-1)

            tau_hat, presum_tau = self.sample_tau(s.size(0) * self.num_q)
            tau_hat = tau_hat.view(self.num_q,
                                   s.size(0), self.num_quantiles)
            presum_tau = presum_tau.view(self.num_q,
                                         s.size(0), self.num_quantiles)
        # [num_q, batch_size, num_quantiles]
        d_vals = self.critic(data, tau_hat)
        # [batch_size, num_quantiles]
        d = torch.quantile(d_vals, self.quantile, dim=0)

        obser_mu, obser_log_std = self.obser_model(
                    torch.cat([s, a, s2, r], dim=-1), d)
        # todo(rui): we can use an optimistic mode to disturb the data/label
        # data = optimistic_policy(data)
        loss_recon_data = self.obser_model.calc_loss(obser_mu, obser_log_std, data)

        '''Reconstruction Q Quantile'''
        q_s, q_a, q_r = self.obser_model.sample(
                *self.obser_model(torch.cat([s, a, s2, r], dim=-1), d.detach()))

        data1 = torch.cat([q_s, a, r], dim=-1)
        data2 = torch.cat([s, q_a, r], dim=-1)
        data3 = torch.cat([s, a, q_r], dim=-1)
        data_ = torch.cat([data1, data2, data3], dim=0)
        tau_hat_ = tau_hat.repeat(1, 3, 1)
        # [num_q, batch_size*3, num_quantiles]
        d_sample = self.critic(data_, tau_hat_)

        with torch.no_grad():
            # [num_q, batch_size*3, 1]
            entropy, norm_weight = self.d_entropy(d_sample, presum_tau.repeat(1, 3, 1))

        obs_info_dict = dict()
        self.attack_infor = (attack_indexes is not None)
        if self.attack_infor:
            obs_info_dict.update(dict(
                attack_entropy = entropy.mean(0)[torch.where(attack_indexes == 1)].mean(-1),
                clean_entropy = entropy.mean(0)[torch.where(attack_indexes == 0)].mean(-1),
                attack_norm_weight = norm_weight.mean(0)[torch.where(attack_indexes == 1)].mean(-1),
                clean_norm_weight = norm_weight.mean(0)[torch.where(attack_indexes == 0)].mean(-1),
            ))

        if self.enable_entropy:
            d_weight = norm_weight
        else:
            d_weight = 1.0

        loss_d = qr_loss(d_sample, d_vals.repeat(1, 3, 1).detach(),
                         tau_hat_, presum_tau.repeat(1, 3, 1),
                         self.sigma, reduction="none")
        loss_recon_d = (loss_d * d_weight).mean()
        return loss_recon_data, loss_recon_d, obs_info_dict

    def update_map(self, s, a, r, s2, nd, attack_indexes, step):
        map_info_dict = dict()

        loss_recon_data, loss_recon_d, obs_info_dict = self.update_obser_model(s, a, r, s2, attack_indexes, step)
        obser_beta = self.obser_beta_decay(step)
        loss_obser_model = loss_recon_data.mean() + loss_recon_d.mean() * obser_beta
        update_params(dict(o=self.obser_optimizer, c=self.critic_optimizer),
                      loss_obser_model)

        loss_recon_data = loss_recon_data.reshape(3, -1)
        map_info_dict.update(obs_info_dict)
        map_info_dict["LossReconState"] = loss_recon_data[0].detach()
        map_info_dict["LossReconAct"] = loss_recon_data[1].detach()
        map_info_dict["LossReconRew"] = loss_recon_data[2].detach()
        map_info_dict["LossReconQval"] = loss_recon_d.detach()
        map_info_dict["InvBeta"] = obser_beta
        return map_info_dict

    def _update(self, data, logger, step, save_log=False):
        self.update_steps += 1

        s, a, s2 = data['state'], data['action'], data['next_state']
        r, nd = data['reward'], data['not_done']
        if r.ndim == 1: r = r.unsqueeze(-1)
        if nd.ndim == 1: nd = nd.unsqueeze(-1)
        attack_indexes = data['attack_indexes'] if 'attack_indexes' in data else None

        '''Calc loss of qfunc and vfunc'''
        q_info_dict, v_info_dict = self.update_critic(s, a, r, s2, nd, attack_indexes, step)

        '''Calc loss of actor'''
        if step % self.actor_update_freq == 0:
            adv = v_info_dict['Adv'].detach()
            pi_info_dict = self.update_actor(s, a, adv)

        '''Maximum A Posterior'''
        map_info_dict = self.update_map(s, a, r, s2, nd, attack_indexes, step)

        '''Save log_info'''
        if save_log:
            self.critic.log(logger['tb'], step, True)
            self.value.log(logger['tb'], step, True)
            self.actor.log(logger['tb'], step, True)

        '''Smooth update'''
        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic, self.critic_targ, self.critic_tau)

        q_info_dict.update(v_info_dict)
        q_info_dict.update(map_info_dict)
        return q_info_dict, pi_info_dict


    def _save(self, model_dir, step):
        torch.save(self.obser_model.state_dict(), '%s/obser_model_%s.pt' % (model_dir, step))

    def _load(self, model_dir, step):
        self.obser_model.load_state_dict(
                torch.load('%s/obser_model_%s.pt' % (model_dir, step),
                           map_location=self.device))

    def _print_log(self, logger):
        logger.log_tabular('Vals', average_only=True)
        logger.log_tabular('TVals', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('Adv', average_only=True)
        logger.log_tabular('Weight', average_only=True)
        logger.log_tabular('HPi', average_only=True)
        logger.log_tabular('LossBC', average_only=True)
        # if self.attack_infor:
        #     logger.log_tabular('attack_TQvals_diff', average_only=True)
        #     logger.log_tabular('clean_TQvals_diff', average_only=True)
        #     logger.log_tabular('attack_TQvals_std', average_only=True)
        #     logger.log_tabular('clean_TQvals_std', average_only=True)
        logger.log_tabular('LossReconAct', average_only=True)
        logger.log_tabular('LossReconQval', average_only=True)
        logger.log_tabular('InvBeta', average_only=True)
