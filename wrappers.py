import math
from collections import deque
from time import perf_counter
from typing import List, Mapping
from citylearn.citylearn import CityLearnEnv

import gym
import numpy as np
from gym import ObservationWrapper, spaces, ActionWrapper
from gym.wrappers import TimeLimit as GymTimeLimit
from gym.wrappers import Monitor as GymMonitor


class RecordEpisodeStatistics(gym.Wrapper):
    """ Multi-agent version of RecordEpisodeStatistics gym wrapper"""

    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.t0 = perf_counter()
        self.episode_reward = np.zeros(self.n_agents)
        self.episode_length = 0
        self.reward_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.episode_reward = 0
        self.episode_length = 0
        self.t0 = perf_counter()

        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self.episode_reward += np.array(reward, dtype=np.float64)
        self.episode_length += 1
        if all(done):
            info["episode_reward"] = self.episode_reward
            for i, agent_reward in enumerate(self.episode_reward):
                info[f"agent{i}/episode_reward"] = agent_reward
            info["episode_length"] = self.episode_length
            info["episode_time"] = perf_counter() - self.t0

            self.reward_queue.append(self.episode_reward)
            self.length_queue.append(self.episode_length)
        return observation, reward, done, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple([
            spaces.flatten(obs_space, obs)
            for obs_space, obs in zip(self.env.observation_space, observation)
        ])


class FlattenAction(ActionWrapper):
    r"""Action wrapper that flattens the actions of individual agents."""

    def __init__(self, env):
        super(FlattenAction, self).__init__(env)

        ma_spaces = []

        for sa_act in env.action_space:
            flatdim = spaces.flatdim(sa_act)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.action_space = spaces.Tuple(tuple(ma_spaces))

    def action(self, action):
        return tuple([
            spaces.flatten(act_space, act)
            for act_space, act in zip(self.env.action_space, action)
        ])


class SquashDones(gym.Wrapper):
    r"""Wrapper that squashes multiple dones to a single one using all(dones)"""

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, all(done), info


class GlobalizeReward(gym.RewardWrapper):
    def reward(self, reward):
        return self.n_agents * [sum(reward)]


class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            # info['TimeLimit.truncated'] = not all(done)
            done = len(observation) * [True]
        return observation, reward, done, info

class ClearInfo(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, {}


class Monitor(GymMonitor):
    def _after_step(self, observation, reward, done, info):
        if not self.enabled: return done

        if done and self.env_semantics_autoreset:
            # For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
            self.reset_video_recorder()
            self.episode_id += 1
            self._flush()

        # Record stats
        self.stats_recorder.after_step(observation, sum(reward), done, info)
        # Record video
        self.video_recorder.capture_frame()

        return done


class DiscreteActionWrapperFix(ActionWrapper):
    """Wrapper for action space discretization.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    bin_sizes: List[Mapping[str, int]], optional
        Then number of bins for each active action in each building.
    default_bin_size: int, default = 10
        The default number of bins if `bin_sizes` is unspecified for any active building action.
    """

    def __init__(self, env: CityLearnEnv, bin_sizes: List[Mapping[str, int]] = None, default_bin_size: int = None):
        super().__init__(env)
        self.env: CityLearnEnv
        assert bin_sizes is None or len(bin_sizes) == len(self.env.buildings), 'length of bin_size must equal number of buildings.'
        self.bin_sizes = [{} for _ in self.env.buildings] if bin_sizes is None else bin_sizes
        self.default_bin_size = 10 if default_bin_size is None else default_bin_size
        self.bin_sizes = [
            {a: s.get(a, self.default_bin_size) for a in b.active_actions} 
            for b, s in zip(self.env.buildings, self.bin_sizes)
        ]
        
    @property
    def action_space(self) -> List[spaces.MultiDiscrete]:
        """Returns action space for discretized actions."""

        if self.env.central_agent:
            bin_sizes = []

            for b in self.bin_sizes:
                for _, v in b.items():
                    bin_sizes.append(v)
            
            action_space = [spaces.MultiDiscrete(bin_sizes)]

        else:
            action_space = [spaces.MultiDiscrete(list(b.values())) for b in self.bin_sizes]

        return action_space

    def action(self, actions: List[float]) -> List[List[float]]:
        """Returns undiscretized actions."""

        transformed_actions = []
        actions = np.array([actions]).T

        for i, (cs, ds) in enumerate(zip(self.env.unwrapped.action_space, self.action_space)):

            transformed_actions_ = []
            
            for j, (ll, hl, b) in enumerate(zip(cs.low, cs.high, ds)):

                a = np.linspace(ll, hl, b.n)[actions[i][j]]
                transformed_actions_.append(a)
            
            transformed_actions.append(transformed_actions_)

        return transformed_actions