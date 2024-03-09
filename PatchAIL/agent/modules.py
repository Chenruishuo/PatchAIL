
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
import torch.nn as nn
import torch.nn.functional as F
import torch
import utils


# class AtariQNetwork(SoftQNetwork):
#     def __init__(self, obs_dim, action_dim, args, device='cpu', input_dim=(84, 84)):
#         super(AtariQNetwork, self).__init__(obs_dim, action_dim, args, device)
#         self.frames = 4
#         self.n_outputs = action_dim

#         # CNN modeled off of Mnih et al.
#         self.cnn = nn.Sequential(
#             nn.Conv2d(self.frames, 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU()
#         )

#         self.fc_layer_inputs = self.cnn_out_dim(input_dim)

#         self.fully_connected = nn.Sequential(
#             nn.Linear(self.fc_layer_inputs, 512, bias=True),
#             nn.ReLU(),
#             nn.Linear(512, self.n_outputs))

#     def cnn_out_dim(self, input_dim):
#         return self.cnn(torch.zeros(1, self.frames, *input_dim)
#                         ).flatten().shape[0]

#     def _forward(self, x, *args):
#         cnn_out = self.cnn(x).reshape(-1, self.fc_layer_inputs)
#         return self.fully_connected(cnn_out)
    
    
class DiscreteCritic(nn.Module):
    def __init__(self, repr_dim, n_actions, feature_dim, hidden_dim):
        super().__init__()

        self.fully_connected = nn.Sequential(
            nn.Linear(repr_dim, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, n_actions))
        
        self.trunk = nn.Identity()
        self.apply(utils.weight_init)

    def forward(self, obs):
        q = self.fully_connected(obs)
        return q

class DiscreteActor(nn.Module):
    def __init__(self, repr_dim, n_actions, feature_dim, hidden_dim, critic=None):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, n_actions),)

        self.critic = critic
        self.apply(utils.weight_init)

    def forward(self, obs, return_action=False, *args, **kwargs):
        if self.critic is None:
            h = self.trunk(obs)
            actions = self.policy(h)
            # dist = F.gumbel_softmax(actions, tau=1, hard=False)
        else:
            actions = self.critic(obs)
            
        dist = utils.MultiNomial(actions)

        if return_action:
            return actions
        
        return dist

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2
