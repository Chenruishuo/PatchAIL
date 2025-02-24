
# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pyrsistent import s
import hydra
import numpy as np
from torch import autograd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

import utils
from agent.modules import Actor, Critic, DiscreteActor, DiscreteCritic
from agent.encoder import Encoder, AtariEncoder
from agent.discriminator import PatchDiscriminator, SmallPatchDiscriminator, BigPatchDiscriminator,\
     IIDPatchDiscriminator, VitDiscriminator, WeightedFeatureDiscriminator, Discriminator, DiscTrunk, AtariPatchDiscriminator
import time
import copy
import gc
from scipy import stats


def compute_gradient_penalty(discriminator, expert_data, policy_data, grad_pen_weight=10.0):
    if len(expert_data.shape) == 2:
        alpha = torch.rand(expert_data.size(0), 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)
    elif len(expert_data.shape) == 4:
        alpha = torch.rand(expert_data.size(0), 1, 1, 1, device=expert_data.device)

    mixup_data = alpha * expert_data + (1 - alpha) * policy_data
    mixup_data.requires_grad = True

    disc = discriminator(mixup_data)
    ones = torch.ones(disc.size()).to(disc.device)
    if len(expert_data.shape) == 2:
        grad = autograd.grad(outputs=disc,
                            inputs=mixup_data,
                            grad_outputs=ones,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True)[0]
    elif len(expert_data.shape) == 4:
        grads = autograd.grad(
                outputs=disc.sum(),
                inputs=mixup_data,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
        grad = grads.view(len(grads[0]), -1)

    grad_pen = grad_pen_weight * (grad.norm(2, dim=1) - 1).pow(2).sum()
    return grad_pen

class DACAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 augment, use_actions, suite_name, obs_type, bc_weight_type, bc_weight_schedule, eta, mix_td,
                 n_actions=None, reward_type="airl", disc_type="encoder", reward_aggr="mean", sim_type="weight",
                 share_encoder=True, state_trans=False, disc_final_iid=False,  disc_aug="random_shift",
                 reward_scale=1.0, grad_pen_weight=10.0, target_disc=False, disc_target_tau=0.05, disc_lr=None,
                 target_enc=False, enc_target_tau=0.05, init_bc_weight=0.933, use_simreg=False, sim_rate=1.5, use_per=False):
        self.device = device
        self.lr = lr
        self._eta = eta
        self.mix_td = mix_td
        self.critic_target_tau = critic_target_tau
        self.disc_target_tau = disc_target_tau
        self.enc_target_tau = enc_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.use_actions = use_actions
        self.use_encoder = True if obs_type=='pixels' else False
        self.target_disc = target_disc
        self.target_enc = target_enc
        self.augment = augment and self.use_encoder
        self.bc_weight_type = bc_weight_type
        self.sim_rate = sim_rate
        self.bc_weight_schedule = bc_weight_schedule
        self.init_bc_weight = init_bc_weight
        self.use_simreg = use_simreg
        self.sim_type = sim_type
        self.use_per = use_per
        if disc_lr is None:
            disc_lr = lr
        if use_simreg:
            print("\n Using Sim Reg!!! Sim Rate: {}".format(sim_rate))
        if target_disc:
            print("\n Using Target Disc!!!")

        self.reward_type = reward_type
        self.disc_type = disc_type
        self.reward_aggr = reward_aggr
        self.share_encoder = share_encoder
        self.state_trans = state_trans
        self.reward_scale = reward_scale
        self.grad_pen_weight = grad_pen_weight

        self.suite_name = suite_name

        self.global_step = 0

        assert disc_type in [
            "encoder",
            "bc_encoder",
            "patch",
            "small_patch",
            "big_patch",
            "input_patch",
            "iid_patch",
            "weighted_feature",
            "vit",
        ], "Invalid disc type!"

        assert reward_type in [
            "airl",
            "gail",
            "fairl",
            "gail2",
        ], "Invalid adversarial irl reward type!"

        assert disc_aug in [
            "random_shift",
            "random_crop",
            "random_cutout",
            "random_aug",
        ], "Invalid discriminator augmentation type!"

        assert sim_type in [
            "weight",
            "bonus",
        ], "Invalid sim type!"

        print("Using reward scale: {}\n".format(reward_scale))
        print("Using reward aggregation : {}\n".format(reward_aggr))
        disc_final_iid = disc_final_iid
        if disc_final_iid:
            print("Using disc final iid: {}\n".format(disc_final_iid))

        # models
        self.encoder = None
        if self.use_encoder:
            if self.suite_name == "atari":
                self.encoder = AtariEncoder(obs_shape).to(device)
                self.encoder_target = AtariEncoder(obs_shape).to(device)
            else:
                self.encoder = Encoder(obs_shape).to(device)
                self.encoder_target = Encoder(obs_shape).to(device)

            repr_dim = self.encoder.repr_dim
            
            self.disc_encoder = self.encoder
            if not self.share_encoder:
                print("No share encoder!!!")
                if self.suite_name == "atari":
                    self.disc_encoder = AtariEncoder(obs_shape).to(device)
                else:
                    self.disc_encoder = Encoder(obs_shape).to(device)
        else:
            repr_dim = obs_shape[0]

        disc_dim = repr_dim + action_shape[0] if use_actions else repr_dim
        disc_dim = repr_dim * 2 if state_trans else disc_dim # if do state trans (s,s'), overwrite use_actions
        if "patch" in self.disc_type:
            disc_dim = obs_shape[0]*2 if self.state_trans else obs_shape[0]
            self.discriminator = PatchDiscriminator(disc_dim, disc_final_iid).to(device)
            if "small" in  self.disc_type:
                self.discriminator = SmallPatchDiscriminator(disc_dim, disc_final_iid).to(device)
            elif "big" in  self.disc_type:
                self.discriminator = BigPatchDiscriminator(disc_dim, disc_final_iid).to(device)
            elif "iid" in  self.disc_type:
                self.discriminator = IIDPatchDiscriminator(disc_dim, disc_final_iid).to(device)

            if self.suite_name == "atari":
                self.discriminator = AtariPatchDiscriminator(disc_dim, disc_final_iid).to(device)

            if self.target_disc:
                self.discriminator_target = self.discriminator.__class__(disc_dim, disc_final_iid).to(device)
                self.discriminator_target.load_state_dict(self.discriminator.state_dict())
        elif "vit" in self.disc_type:
            disc_dim = obs_shape[0]*2 if self.state_trans else obs_shape[0]
            self.discriminator = VitDiscriminator(disc_dim).to(device)
        elif "weighted_feature" in self.disc_type:
            disc_dim = 64 if self.state_trans else 32
            self.discriminator = WeightedFeatureDiscriminator(disc_dim).to(device)
            self.encoder.unflatten = True
        else:
            self.discriminator = Discriminator(disc_dim, hidden_dim).to(device)

        if not self.share_encoder and not(("patch" in self.disc_type) or ("vit" in self.disc_type)):
            self.disc_trunk = DiscTrunk(repr_dim, feature_dim).to(device)
        
        if suite_name == "atari":
            self.actor = DiscreteActor(repr_dim, n_actions, feature_dim,
                           hidden_dim).to(device)
        else:
            self.actor = Actor(repr_dim, action_shape, feature_dim,
                            hidden_dim).to(device)
        if suite_name == "atari":
            self.critic = DiscreteCritic(repr_dim, n_actions, feature_dim,
                             hidden_dim).to(device)
            self.critic_target = DiscreteCritic(repr_dim, n_actions,
                                        feature_dim, hidden_dim).to(device)
            self.actor.critic = self.critic
        else:
            self.critic = Critic(repr_dim, action_shape, feature_dim,
                                hidden_dim).to(device)
            self.critic_target = Critic(repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())


        self.encoder_bc = self.encoder
        self.actor_bc = self.actor

        # optimizers
        if self.use_encoder:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.discriminator_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=disc_lr)
        if not self.share_encoder and not(("patch" in self.disc_type) or ("vit" in self.disc_type)):
            self.discriminator_opt = torch.optim.Adam(
            list(self.discriminator.parameters())+list(self.disc_trunk.parameters())+list(self.disc_encoder.parameters()), lr=disc_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = utils.RandomShiftsAug(pad=4)
        
        if disc_aug == "random_shift":
            self.disc_aug = self.aug
        elif disc_aug == "random_crop":
            self.disc_aug = utils.RandomCropAug() 
        elif disc_aug == "random_cutout":
            self.disc_aug = utils.RandomCutAug()
        elif disc_aug == "random_aug":
            self.disc_aug = utils.RandomAug()
        else:
            raise NotImplementedError
        
        if not self.augment:
            self.aug = lambda x: x
            self.disc_aug = lambda x: x

        print("Using disc aug: {}\n".format(disc_aug))

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        if self.use_encoder:
            self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.discriminator.train(training)

    def __repr__(self):
        return 'dac'

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)

        obs = self.encoder(obs.unsqueeze(0)) if self.use_encoder else obs.unsqueeze(0)
        if "weighted_feature" in self.disc_type:
            obs, _ = obs
        stddev = utils.schedule(self.stddev_schedule, step)

        dist = self.actor(obs, std=stddev)

        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                try:
                    action.uniform_(-1.0, 1.0)
                except:
                    action = dist.uniform()
        return action.cpu().numpy()[0]

    def compute_td_error(self, obs, action, reward, discount, next_obs):
        obs, action, reward, discount, next_obs = utils.to_torch((obs, action, reward, discount, next_obs), self.device)
        obs = self.encoder(obs.unsqueeze(0))
        next_obs = self.encoder(next_obs.unsqueeze(0))

        assert self.suite_name == "atari", "current not tested for tasks other than atari"
        with torch.no_grad():
            dist = self.critic(next_obs)
            next_action = dist.argmax(dim=-1)
            target_Q = self.critic_target(next_obs)[range(len(obs)),next_action]
            target_Q = reward + (discount * target_Q)

            Q = self.critic(obs)[range(len(obs)),action.long()]

        return np.absolute((target_Q - Q).detach().cpu().numpy())

    def update_discrete_critic(self, obs, action, initial_obs, initial_action, reward, expert_reward, discount, next_obs, initial_next_obs, bc_regularize, step, expert_obs, expert_action, expert_next_obs, **kwargs):
        metrics = dict()
        half_batch_size = obs.shape[0]//2
        if self.mix_td:
            obs = torch.cat([obs[:half_batch_size],expert_obs[:half_batch_size]],dim=0)
            action = torch.cat([action[:half_batch_size], expert_action[:half_batch_size]],dim=0)
            next_obs = torch.cat([next_obs[:half_batch_size],expert_next_obs[:half_batch_size]],dim=0)
            reward = torch.cat((reward[:half_batch_size], expert_reward[:half_batch_size]),dim=0)
        with torch.no_grad():
            dist = self.critic(next_obs)
            next_action = dist.argmax(dim=-1)
            target_Q = self.critic_target(next_obs)[range(len(obs)),next_action].unsqueeze(-1)
            target_Q = reward + (discount * target_Q)

        Q = self.critic(obs)[range(len(obs)),action.long()].unsqueeze(-1)
        
        is_weights = 1.0
        if self.use_per:
            metrics["tree_indices"] = kwargs["tree_indices"].cpu().numpy()
            metrics["td_errors"] = np.absolute((target_Q - Q).squeeze().detach().cpu().numpy())
            is_weights = kwargs["is_weights"].unsqueeze(-1)
        
        # Compute bc weight
        if not bc_regularize:
            bc_weight = 0.0
        elif self.bc_weight_type == "linear":
            bc_weight = utils.schedule(self.bc_weight_schedule, step)
        elif self.bc_weight_type == "exponential":
            bc_weight = self.init_bc_weight ** (step / 500)
        elif self.bc_weight_type == "qfilter":
            raise NotImplementedError
        
        bc_loss = 0.0
        if bc_regularize:
            logit_bc = self.critic(expert_obs)
            bc_loss = nn.CrossEntropyLoss()(logit_bc, expert_action)
        
        obs_to_max = torch.cat([initial_obs[:half_batch_size], expert_obs[:half_batch_size]],dim=0)
        dist_to_max = self.critic(obs_to_max)
        softmax_to_max = F.softmax(dist_to_max, dim=-1)
        with torch.no_grad():
            pi_action_to_max = torch.multinomial(softmax_to_max, 1)
        pi_q_to_max = torch.gather(softmax_to_max, -1, pi_action_to_max)

        delta_Q = -pi_q_to_max
        critic_loss = (is_weights * F.mse_loss(Q, target_Q, reduction='none')).mean()*(1-bc_weight) - bc_loss*bc_weight*0.03

        optimistic_loss = self._eta * delta_Q.mean()

        total_loss = critic_loss + optimistic_loss

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q'] = Q.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['optimistic_loss'] = optimistic_loss.item()
        if bc_regularize:
            metrics['bc_loss'] = bc_loss.item()

        # optimize encoder and critic
        if self.use_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        self.critic_opt.step()
        if self.use_encoder:
            self.encoder_opt.step()

        return metrics

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)

            dist = self.actor(next_obs, std=stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)

        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        if self.use_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.use_encoder:
            self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, expert_obs, obs_qfilter, expert_action, bc_regularize, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)

        dist = self.actor(obs, std=stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        # Compute bc weight
        if not bc_regularize:
            bc_weight = 0.0
        elif self.bc_weight_type == "linear":
            bc_weight = utils.schedule(self.bc_weight_schedule, step)
        elif self.bc_weight_type == "exponential":
            bc_weight = self.init_bc_weight ** (step / 500)
        elif self.bc_weight_type == "qfilter":
            """
            Soft Q-filtering inspired from             
            Nair, Ashvin, et al. "Overcoming exploration in reinforcement 
            learning with demonstrations." 2018 IEEE international 
            conference on robotics and automation (ICRA). IEEE, 2018.
            """
            with torch.no_grad():
                stddev = 0.1
                dist_qf = self.actor_bc(obs_qfilter, std=stddev)
                action_qf = dist_qf.mean
                Q1_qf, Q2_qf = self.critic(obs_qfilter.clone(), action_qf)
                Q_qf = torch.min(Q1_qf, Q2_qf)
                bc_weight = (Q_qf>Q).float().mean().detach()

        actor_loss = - Q.mean() * (1-bc_weight)

        stddev = 0.1
        dist_bc = self.actor(expert_obs, std=stddev)
        log_prob_bc = dist_bc.log_prob(expert_action).sum(-1, keepdim=True)
        if bc_regularize:
            actor_loss = - Q.mean() * (1-bc_weight) - log_prob_bc.mean()*bc_weight*0.03

        # optimize actor
        torch.autograd.set_detect_anomaly(True)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            metrics['actor_q'] = Q.mean().item()
            if bc_regularize and self.bc_weight_type == "qfilter":
                metrics['actor_qf'] = Q_qf.mean().item()
            metrics['bc_weight'] = bc_weight
            metrics['regularized_rl_loss'] = -Q.mean().item()* (1-bc_weight)
            metrics['rl_loss'] = -Q.mean().item()
            if bc_regularize:
                metrics['regularized_bc_loss'] = - log_prob_bc.mean().item()*bc_weight*0.03
                metrics['bc_loss'] = - log_prob_bc.mean().item()*0.03
            
        return metrics

    def update(self, replay_iter, expert_replay_iter, initial_iter, step, bc_regularize=False, expert_demo=None, update_disc=True):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        self.global_step = step
        
        obs, action, reward, discount, next_obs = utils.to_torch(next(replay_iter), self.device)
        reward = self.dac_rewarder(obs, action, next_obs)
        expert_obs, expert_action, expert_next_obs = utils.to_torch(next(expert_replay_iter), self.device)
        expert_reward = self.dac_rewarder(expert_obs, expert_action, expert_next_obs, clip=(self.suite_name=='atari'))
        initial_obs, initial_action, initial_next_obs = utils.to_torch(next(initial_iter),self.device)

        obs = obs.float()
        next_obs = next_obs.float()
        expert_obs = expert_obs.float()
        expert_next_obs = expert_next_obs.float()
        initial_obs = initial_obs.float()
        initial_next_obs = initial_next_obs.float()

        # augment
        if self.use_encoder and self.augment:
            obs = self.aug(obs)
            next_obs = self.aug(next_obs)
            expert_obs = self.aug(expert_obs)
            expert_next_obs = self.aug(expert_next_obs) 
            initial_obs = self.aug(initial_obs)
            initial_next_obs = self.aug(initial_next_obs)

        if self.use_encoder:
            obs = self.encoder(obs)
            next_obs = self.encoder(next_obs)
            expert_obs = self.encoder(expert_obs)
            expert_next_obs = self.encoder(expert_next_obs)
            initial_obs = self.encoder(initial_obs)
            initial_next_obs = self.encoder(initial_next_obs)
        
        if update_disc:
            results = self.update_discriminator(obs, action, expert_obs, expert_action, next_obs, expert_next_obs)
            metrics.update(results)

        if self.suite_name == "atari":
            metrics.update(self.update_discrete_critic(obs, action, initial_obs, initial_action, reward, expert_reward, discount, next_obs, initial_next_obs, 
                                                       bc_regularize=bc_regularize, step=step, expert_obs=expert_obs, expert_action=expert_action, expert_next_obs=expert_next_obs))
        else:
            # update critic
            metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))
            # update actor
            metrics.update(self.update_actor(obs.detach(), expert_obs, obs, expert_action, bc_regularize, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)
        # update encoder target
        utils.soft_update_params(self.encoder, self.encoder_target,
                                 self.enc_target_tau)
        
        if self.target_disc:
            # update dics target
            utils.soft_update_params(self.discriminator, self.discriminator_target,
                                    self.disc_target_tau)

        metrics.update(self.record_grad_norm(self.critic, "critic"))
        if self.suite_name != "atari":
            metrics.update(self.record_grad_norm(self.actor, "actor"))
        if update_disc:
            metrics.update(self.record_grad_norm(self.discriminator, "discriminator"))
        metrics.update(self.record_grad_norm(self.encoder, "encoder"))

        return metrics

    def record_grad_norm(self, model, net_name):
        """
        Record the grad norm for monitoring.
        """
        metrics = dict()
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        metrics[net_name+"grad_norm"] = total_norm

        return metrics
        
    def dac_rewarder(self, obses, actions=None, next_obses=None, clip=False):
        if type(obses) == np.ndarray:
            obses = torch.tensor(obses).to(self.device)
        if type(next_obses) == np.ndarray:
            next_obses = torch.tensor(next_obses).to(self.device)
        obses = self.critic.trunk(self.encoder(obses)) if self.share_encoder else self.disc_trunk(self.disc_encoder(obses))
        if next_obses is not None:
            next_obses = self.critic.trunk(self.encoder(next_obses)) if self.share_encoder else self.disc_trunk(self.disc_encoder(next_obses))
        if self.use_actions:
            assert actions is not None, "actions should not be None!"
            actions = torch.tensor(actions).to(self.device)
            obses = torch.cat([obses, actions], dim=1)
        if self.state_trans:
            if next_obses is not None:
                obses = torch.cat([obses, next_obses], dim=1)
            else:
                obses = torch.cat([obses[0].unsqueeze(0), obses]) # for dummy first state
                obses = torch.cat([obses[:-1], obses[1:]], dim=1)
        with torch.no_grad():
            with utils.eval_mode(self.discriminator):
                reward = self.discriminator(obses)
        if clip:
                reward = torch.clamp(reward, min=-10, max=10)
        return reward
        

    def update_discriminator(self, policy_obs, policy_action, expert_obs,
                             expert_action, policy_next_obs=None, expert_next_obs=None):
        metrics = dict()
        batch_size = expert_obs.shape[0]//2
        obs_shape = expert_obs.shape[1]
        # policy batch size is 2x
        expert_obs = expert_obs[:batch_size]
        expert_next_obs = expert_next_obs[:batch_size]
        expert_action = expert_action[:batch_size]
        policy_obs = policy_obs[:batch_size]
        policy_next_obs = policy_next_obs[:batch_size]
        policy_action = policy_action[:batch_size]

        ones = torch.ones(batch_size, device=self.device)
        zeros = torch.zeros(batch_size, device=self.device)

        disc_obs = disc_input = torch.cat([expert_obs, policy_obs], dim=0)

        if self.state_trans: # D(s,s')
            disc_next_obs = torch.cat([expert_next_obs, policy_next_obs], dim=0)
            disc_input = torch.cat([disc_input, disc_next_obs], dim=1) # This is for PatchIRL
        else: # overwrite use_actions if state_trans
            if self.use_actions: # D(s,a)
                disc_action = torch.cat([expert_action, policy_action], dim=0)
                disc_input = torch.cat([disc_input, disc_action], dim=1)

        disc_label = torch.cat([ones, zeros], dim=0).unsqueeze(dim=1)
        
        if ("weighted_feature" not in self.disc_type) and ("patch" not in self.disc_type) and ("vit" not in self.disc_type) and self.use_encoder:
            if self.state_trans:
                disc_obs = self.critic.trunk(disc_obs) if self.share_encoder else self.disc_trunk(disc_obs)
                disc_next_obs = self.critic.trunk(disc_next_obs) if self.share_encoder else self.disc_trunk(disc_next_obs)
                disc_input = torch.cat([disc_obs, disc_next_obs], dim=1) # get (s,s') for EncIRL
            else: # get (s) or (s,a) for EncIRL
                disc_input = self.critic.trunk(disc_input) if self.share_encoder else self.disc_trunk(disc_input)

        if (("patch" not in self.disc_type) and ("vit" not in self.disc_type)) and self.share_encoder:
            disc_input = disc_input.detach() # Do not update the encoder if using shared encoder

        disc_output = self.discriminator(disc_input)
        patch_number = 1
        if disc_label.shape != disc_output.shape: # this is for patch gail - (B, P_W, P_H, 1)
            disc_output = disc_output.view(disc_output.shape[0],-1)
            patch_number = disc_output.shape[1]
            disc_label = disc_label.expand_as(disc_output)

        dac_loss = F.binary_cross_entropy_with_logits(disc_output,
                                                      disc_label,
                                                      reduction='sum')

        expert_obs, policy_obs = torch.split(disc_input, batch_size, dim=0)
        grad_pen = compute_gradient_penalty(self.discriminator, expert_obs.detach(),
                                            policy_obs.detach(), self.grad_pen_weight)

        dac_loss /= (batch_size * patch_number)
        grad_pen /= (batch_size * patch_number)

        metrics['disc_loss'] = dac_loss.mean().item()
        metrics['disc_grad_pen'] = grad_pen.mean().item()

        self.discriminator_opt.zero_grad(set_to_none=True)
        dac_loss.backward()
        grad_pen.backward()
        self.discriminator_opt.step()
        return metrics

    def save_snapshot(self):
        keys_to_save = ['actor', 'critic', 'discriminator']
        if self.use_encoder:
            keys_to_save += ['encoder']
            if not self.share_encoder and not(("patch" in self.disc_type) or ("vit" in self.disc_type)):
                keys_to_save += ['disc_encoder']
                keys_to_save += ['disc_trunk']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        return payload

    def load_snapshot(self, payload):
        for k, v in payload.items():
            self.__dict__[k] = v
        self.critic_target.load_state_dict(self.critic.state_dict())
        if self.use_encoder:
            self.encoder_target.load_state_dict(self.encoder.state_dict())
        
        if self.bc_weight_type == "qfilter":
            # Store a copy of the BC policy with frozen weights
            if self.use_encoder:
                self.encoder_bc = copy.deepcopy(self.encoder)
                for param in self.encoder_bc.parameters():
                    param.requires_grad = False
            self.actor_bc = copy.deepcopy(self.actor)
            for param in self.actor_bc.parameters():
                param.required_grad = False

        if self.use_encoder and self.share_encoder:
            self.disc_encoder = self.encoder

        # Update optimizers
        if self.use_encoder:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)