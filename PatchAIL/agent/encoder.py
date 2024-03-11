
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
import torch
import utils


class Encoder(nn.Module):
    def __init__(self, obs_shape, input_dim=(84, 84)):
        super().__init__()

        assert len(obs_shape) == 3
        self.frames = 4
        self.cnn = nn.Sequential(
            nn.Conv2d(self.frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.repr_dim = self.cnn_out_dim(input_dim)

        self.apply(utils.weight_init)
    
    def cnn_out_dim(self, input_dim):
        return self.cnn(torch.zeros(1, self.frames, *input_dim)
                        ).flatten().shape[0]

    def forward(self, x):
        cnn_out = self.cnn(x).reshape(-1, self.repr_dim)
        return cnn_out


class AtariEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.unflatten = False
        
        # CNN modeled off of Mnih et al.
        self.repr_dim = 3136
        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 8, stride=4),
                                     nn.ReLU(), nn.Conv2d(32, 64, 4, stride=2),
                                     nn.ReLU(), nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU())
        
        # self.trunk = nn.Sequential(nn.Linear(self.repr_dim, feature_dim),
        #                            nn.LayerNorm(feature_dim), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h_flat = h.view(h.shape[0], -1)
        return h_flat

class EasyEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 225792

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=1, padding='same'),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1, padding='same'),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1, padding='same'),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1, padding='same'),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        return h