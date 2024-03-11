#!/usr/bin/env python3

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

from time import time
import warnings
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'osmesa'
from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_expert_replay_loader,InitialBuffer,make_initial_loader
from video import TrainVideoRecorder, VideoRecorder
import pickle
import random

warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec[cfg.obs_type].shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


def add_noise(observation, zone='random', noise_size=14, noise_type='random', noise_scale=0.1):
    if zone is None:
        return observation
    c, h, w = observation.shape
    if zone == 'random':
        noise_begin_h = random.randint(0, h - noise_size - 1)
        noise_begin_w = random.randint(0, w - noise_size - 1)
        observation[:, noise_begin_h:noise_begin_h+noise_size, noise_begin_w:noise_begin_w+noise_size] = observation[:, noise_begin_h:noise_begin_h+noise_size, noise_begin_w:noise_begin_w+noise_size] // 2 +\
             np.random.randint(low=0, high=255, size=(c, noise_size, noise_size), dtype=np.uint8) // 2
    elif zone == 'left-top':
        observation[:, :noise_size, :noise_size] = np.random.randint(low=0, high=255, size=(c, noise_size, noise_size), dtype=np.uint8) # // 2 + observation[:, :noise_size, :noise_size] // 2
    else:
        raise NotImplementedError

    return observation  
    

class WorkspaceIL:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()
        
        if cfg.suite.name == 'atari':
            cfg.agent.n_actions = self.train_env.action_spec().num_values

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(), cfg.agent)

        if repr(self.agent) == 'bc':
            self.cfg.suite.num_train_frames = self.cfg.num_train_frames_bc
            self.cfg.suite.num_seed_frames = 0

        self.expert_replay_loader = make_expert_replay_loader(
            self.cfg.expert_dataset, self.cfg.batch_size, self.cfg.num_demos, self.cfg.obs_type)
        self.expert_replay_iter = iter(self.expert_replay_loader)
            
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        self.noise_zone = None
        self.noise_scale = 0.1
        self.noise_size = 3

        with open(self.cfg.expert_dataset, 'rb') as f:
            if self.cfg.obs_type == 'pixels':
                self.expert_demo, _, _, self.expert_reward = pickle.load(f)
            elif self.cfg.obs_type == 'features':
                _, self.expert_demo, _, self.expert_reward = pickle.load(f)
        self.expert_demo = self.expert_demo[:self.cfg.num_demos]
        print(np.mean([np.sum(_) for _ in self.expert_reward]), np.std([np.sum(_) for _ in self.expert_reward]))
        self.expert_reward = np.mean([np.mean(_) for _ in self.expert_reward[:self.cfg.num_demos]])

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.train_env = hydra.utils.call(self.cfg.suite.task_make_fn)
        self.eval_env = hydra.utils.call(self.cfg.suite.task_make_fn)

        # create replay buffer
        data_specs = [
            self.train_env.observation_spec()[self.cfg.obs_type],
            self.train_env.action_spec(),
            specs.Array((1, ), np.float32, 'reward'),
            # specs.Array((1521, ), np.float32, 'reward'),
            specs.Array((1, ), np.float32, 'discount')
        ]

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader, self.replay_buf = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.suite.save_snapshot, self.cfg.nstep, self.cfg.suite.discount)
        
        self.initial_buffer = InitialBuffer(
                self.cfg.expert_dataset, self.cfg.num_demos, self.cfg.obs_type, self.cfg.nstep)
        self.initial_loader = make_initial_loader(self.initial_buffer, self.cfg.batch_size,
                                                  self.cfg.initial_buffer_num_workers)

        self._replay_iter = None
        self.expert_replay_iter = None
        self.video_recorder = None
        self.train_video_recorder = None
        self._initial_iter = None
        
        if self.cfg.record_video:
            self.video_recorder = VideoRecorder(
                self.work_dir if self.cfg.save_video else None)
            self.train_video_recorder = TrainVideoRecorder(
                self.work_dir if self.cfg.save_train_video else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter
    
    @property
    def initial_iter(self):
        if self._initial_iter is None:
            self._initial_iter = iter(self.initial_loader)
        return self._initial_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)

        if self.cfg.suite.name == 'openaigym' or self.cfg.suite.name == 'metaworld':
            paths = []
        while eval_until_episode(episode):
            if self.cfg.suite.name == 'metaworld':
                path = []
            time_step = self.eval_env.reset()
            if self.video_recorder:
                self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                # + Noise
                time_step.observation[self.cfg.obs_type] = add_noise(time_step.observation[self.cfg.obs_type], zone=self.noise_zone, noise_scale=self.noise_scale, noise_size=self.noise_size)
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation[self.cfg.obs_type],
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                if self.cfg.suite.name == 'metaworld':
                    path.append(time_step.observation['goal_achieved'])
                if self.video_recorder:
                    self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            if self.video_recorder:
                self.video_recorder.save(f'{self.global_frame}.mp4')
        
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.suite.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train_il(self):
        # predicates
        train_until_step = utils.Until(self.cfg.suite.num_train_frames,
                                       self.cfg.suite.action_repeat)
        seed_until_step = utils.Until(self.cfg.suite.num_seed_frames,
                                      self.cfg.suite.action_repeat)
        eval_every_step = utils.Every(self.cfg.suite.eval_every_frames,
                                      self.cfg.suite.action_repeat)

        episode_step, episode_reward = 0, 0

        if self.noise_zone is not None:
            print("\n Using Noisy Observation! \n")

        time_steps = list()
        observations = list()
        actions = list()

        time_step = self.train_env.reset()
        time_steps.append(time_step)
        # + Noise
        time_step.observation[self.cfg.obs_type] = add_noise(time_step.observation[self.cfg.obs_type], zone=self.noise_zone, noise_scale=self.noise_scale, noise_size=self.noise_size)
        observations.append(time_step.observation[self.cfg.obs_type])
        actions.append(time_step.action)
        
        if self.train_video_recorder:
            self.train_video_recorder.init(time_step.observation[self.cfg.obs_type])
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                if self._global_episode % 10 == 0:
                    if self.train_video_recorder:
                        self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                observations = np.stack(observations, 0)
                actions = np.stack(actions, 0)

                # Set new rewards
                new_rewards = self.agent.dac_rewarder(observations, actions).flatten().detach().cpu().numpy()

                new_rewards_sum = np.sum(new_rewards)
                if len(new_rewards.shape) >= 2:
                    new_rewards_sum = np.sum(new_rewards.mean(axis=1))

                for i, elt in enumerate(time_steps):
                    elt = elt._replace(
                        observation=time_steps[i].observation[self.cfg.obs_type])
                    self.replay_storage.add(elt)

                # update initial buffer
                self.initial_buffer.store_initial((
                    time_steps[0].observation[self.cfg.obs_type],
                    time_steps[1].action,
                    time_steps[self.cfg.nstep].observation[self.cfg.obs_type]
                ))

                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.suite.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('initial_buffer_size', self.initial_buffer.size())
                        log('seed',self.cfg.seed)
                        log('step', self.global_step)
                        if repr(self.agent) == 'potil' or repr(self.agent) == 'dac':
                                # log('expert_reward', self.expert_reward)
                                log('imitation_reward', new_rewards_sum)

                # reset env
                time_steps = list()
                observations = list()
                actions = list()

                time_step = self.train_env.reset()
                time_steps.append(time_step)
                # + Noise
                time_step.observation[self.cfg.obs_type] = add_noise(time_step.observation[self.cfg.obs_type], zone=self.noise_zone, noise_scale=self.noise_scale, noise_size=self.noise_size)
                observations.append(time_step.observation[self.cfg.obs_type])
                actions.append(time_step.action)
                if self.train_video_recorder:
                    self.train_video_recorder.init(time_step.observation[self.cfg.obs_type])
                # try to save snapshot
                if self.cfg.suite.save_snapshot and (self.global_step % 100 == 0):
                    if "regular_save" in self.cfg and self.cfg.regular_save:
                        self.save_snapshot(self.global_step)
                    else:
                        self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()
                
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation[self.cfg.obs_type],
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                # Update
                metrics = self.agent.update(self.replay_iter, self.expert_replay_iter, self.initial_iter, self.global_step)
                
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward

            time_steps.append(time_step)
            observations.append(time_step.observation[self.cfg.obs_type])
            actions.append(time_step.action)
            
            if self.train_video_recorder:
                self.train_video_recorder.record(time_step.observation[self.cfg.obs_type])
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self, suffix=''):
        snapshot = self.work_dir / 'snapshot{}.pt'.format(suffix)
        keys_to_save = ['timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, snapshot):
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
            else:
                self.__dict__[k] = v
        self.agent.load_snapshot(agent_payload)

def delete_file(dir_path, name):
    for f in dir_path.glob(name):
        f.unlink()

@hydra.main(config_path='cfgs', config_name='config_normal')
def main(cfg):
    utils.set_seed_everywhere(cfg.seed)
    torch.backends.cudnn.deterministic=True
    from train import WorkspaceIL as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    
    # Load weights
    if 'resume_exp' in cfg and cfg.resume_exp:
        print(f'resuming exp')
        snapshot = workspace.work_dir / 'snapshot.pt'
        if snapshot.exists():
            print(f'load from snapshot: {snapshot}')
            workspace.load_snapshot(snapshot)
        else:
            delete_file(workspace.work_dir, 'buffer/*.npz')
            delete_file(workspace.work_dir, 'tb/*')
            delete_file(workspace.work_dir, '*.csv')
    else: # delete all buffer files
        print(f'start a new exp')
        delete_file(workspace.work_dir, 'buffer/*.npz')
        delete_file(workspace.work_dir, 'tb/*')
        delete_file(workspace.work_dir, '*.csv')

    workspace.train_il()


if __name__ == '__main__':
    main()
