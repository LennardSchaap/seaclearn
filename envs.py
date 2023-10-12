import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box
from citylearn.citylearn import CityLearnEnv

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper

class MADummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        agents = len(self.observation_space)
        # change this because we want >1 reward
        self.buf_rews = np.zeros((self.num_envs, agents), dtype=np.float32)

def make_env(dataset_name, time_limit):
    def _thunk():

        dataset_name = 'citylearn_challenge_2022_phase_1'
        env = CityLearnEnv(dataset_name, central_agent=False, time_limit=time_limit)

        return env

    return _thunk

def make_vec_envs(dataset_name, parallel, time_limit, device):
    envs = [
        make_env(dataset_name, time_limit) for i in range(parallel)
    ]

    envs = SubprocVecEnv(envs, start_method="fork")

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

