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

        rewards = []

        for o in observations:
            cost = o['net_electricity_consumption']
            battery_soc = o['electrical_storage_soc']
            penalty = -(1.0 + np.sign(cost) * battery_soc)
            reward = penalty * abs(cost)
            rewards.append(reward)

        return rewards

def set_active_observations(
    schema: dict, active_observations: List[str]
) -> dict:
    """Set the observations that will be part of the environment's
    observation space that is provided to the control agent.

    Parameters
    ----------
    schema: dict
        CityLearn dataset mapping used to construct environment.
    active_observations: List[str]
        Names of observations to set active to be passed to control agent.

    Returns
    -------
    schema: dict
        CityLearn dataset mapping with active observations set.
    """

    active_count = 0

    for o in schema['observations']:
        if o in active_observations:
            schema['observations'][o]['active'] = True
            active_count += 1
        else:
            schema['observations'][o]['active'] = False

    valid_observations = list(schema['observations'].keys())
    assert active_count == len(active_observations),\
        'the provided observations are not all valid observations.'\
          f' Valid observations in CityLearn are: {valid_observations}'

    return schema

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

def _make_env(env_name, rank, time_limit, wrappers, default_bin_size, monitor_dir, random_start = False, evaluate = False):

    def _thunk():

        seed = 123

        start_pos = 0
        end_pos = 7700

        env = CityLearnEnv(env_name, central_agent=False, simulation_start_time_step=start_pos, simulation_end_time_step=end_pos)
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

