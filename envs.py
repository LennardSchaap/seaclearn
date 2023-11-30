import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box
from gym.wrappers import Monitor

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

from citylearn.wrappers import StableBaselines3Wrapper, DiscreteActionWrapper
from citylearn.citylearn import CityLearnEnv
from wrappers import TimeLimit, Monitor, DiscreteActionWrapperFix

class MADummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        agents = len(self.observation_space)
        # change this because we want >1 reward
        self.buf_rews = np.zeros((self.num_envs, agents), dtype=np.float32)

def make_env(env_name, rank, time_limit, wrappers, default_bin_size, monitor_dir, random_start = False, evaluate = False):

    start_pos = 7759
    end_pos = 8759

    env = CityLearnEnv(env_name, central_agent=False, simulation_start_time_step=start_pos, simulation_end_time_step=end_pos)

    for wrapper in wrappers:
        if wrapper == DiscreteActionWrapper or wrapper == DiscreteActionWrapperFix:
            env = wrapper(env, default_bin_size=default_bin_size)
        else:
            env = wrapper(env)

    return env

def _make_env(env_name, rank, time_limit, wrappers, default_bin_size, monitor_dir, random_start = False, evaluate = False):

    def _thunk():

        seed = 123

        start_pos = 0
        end_pos = 1000

        env = CityLearnEnv(env_name, central_agent=False, simulation_start_time_step=start_pos, simulation_end_time_step=end_pos)

        if time_limit:
            env = TimeLimit(env, time_limit)
        for wrapper in wrappers:
            if wrapper == DiscreteActionWrapper or wrapper == DiscreteActionWrapperFix:
                env = wrapper(env, default_bin_size=default_bin_size)
            else:
                env = wrapper(env)
        
        if monitor_dir:
            env = Monitor(env, monitor_dir, lambda ep: int(ep==0), force=True, uid=str(rank))

        return env

    return _thunk

def make_vec_envs(
    env_name, parallel, time_limit, wrappers, default_bin_size, device, monitor_dir=None
):
    envs = [
        _make_env(env_name, i, time_limit, wrappers, default_bin_size, monitor_dir) for i in range(parallel)
    ]

    if len(envs) == 1 or monitor_dir:
        envs = MADummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs, start_method="spawn")

    envs = VecPyTorch(envs, device)
    return envs

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        return [torch.from_numpy(o).to(self.device) for o in obs]
        return obs

    def step_async(self, actions):
        actions = [a.squeeze().cpu().numpy() for a in actions]
        actions = list(zip(*actions))
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        return (
            [torch.from_numpy(o).float().to(self.device) for o in obs],
            torch.from_numpy(rew).float().to(self.device),
            torch.from_numpy(done).float().to(self.device),
            info,
        )

