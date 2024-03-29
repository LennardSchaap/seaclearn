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

from typing import Any, List, Mapping, Union
from citylearn.reward_function import RewardFunction

def scale_and_clip_rewards(rewards, scale_factor=1.0):
    scaled_rewards = [reward * scale_factor for reward in rewards]
    clipped_rewards = [max(-1, min(reward, 1)) for reward in scaled_rewards]
    return clipped_rewards

class CustomReward(RewardFunction):
    def __init__(self, env: CityLearnEnv):
        r"""Initialize CustomReward.

        Parameters
        ----------
        env: Mapping[str, CityLearnEnv]
            CityLearn environment instance.
        """

        super().__init__(env)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        r"""Returns reward for most recent action.

        The reward is designed to minimize electricity cost.
        It is calculated for each building, i and summed to provide the agent
        with a reward that is representative of all n buildings.
        It encourages net-zero energy use by penalizing grid load satisfaction
        when there is energy in the battery as well as penalizing
        net export when the battery is not fully charged through the penalty
        term. There is neither penalty nor reward when the battery
        is fully charged during net export to the grid. Whereas, when the
        battery is charged to capacity and there is net import from the
        grid the penalty is maximized.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.
        """

        electricity_consumption=[o['net_electricity_consumption'] for o in observations]
        carbon_emission=[o['carbon_intensity']*o['net_electricity_consumption'] for o in observations]
        electricity_price=[o['electricity_pricing']*o['net_electricity_consumption'] for o in observations]

        electricity_price = np.array(electricity_price).clip(0.) + (-np.array(electricity_price)).clip(0.) * 0.3
        carbon_emission = np.array(carbon_emission).clip(0.) + (-np.array(carbon_emission)).clip(0.) * 0.3
        reward = -(electricity_price + carbon_emission)

        return reward

class MADummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        agents = len(self.observation_space)
        # change this because we want >1 reward
        self.buf_rews = np.zeros((self.num_envs, agents), dtype=np.float32)

def make_env(env_name, rank, time_limit, wrappers, default_bin_size, monitor_dir, random_start = False, evaluate = False):

    start_pos = 7700
    end_pos = 8759

    env = CityLearnEnv(env_name, central_agent=False, simulation_start_time_step=start_pos, simulation_end_time_step=end_pos)

    for wrapper in wrappers:
        if wrapper == DiscreteActionWrapper or wrapper == DiscreteActionWrapperFix:
            env = wrapper(env, default_bin_size=default_bin_size)
        else:
            env = wrapper(env)

    return env

def _make_env(env_name, rank, time_limit, wrappers, default_bin_size, monitor_dir, random_start = False, evaluate = False, custom_reward = False):

    def _thunk():

        seed = 123

        start_pos = 0
        end_pos = 7700

        env = CityLearnEnv(env_name, central_agent=False, simulation_start_time_step=start_pos, simulation_end_time_step=end_pos)
        if custom_reward:
            env.reward_function = CustomReward(env)

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
    env_name, parallel, time_limit, wrappers, default_bin_size, device, monitor_dir=None, custom_reward = False
):
    envs = [
        _make_env(env_name, i, time_limit, wrappers, default_bin_size, monitor_dir, custom_reward = custom_reward) for i in range(parallel)
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

