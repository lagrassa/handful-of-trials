from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import spaces, utils
from gym.envs.robotics import FetchPushEnv


class NongoalFetchPushEnv(FetchPushEnv):

    def __init__(self, *args, **kwargs):
        FetchPushEnv.__init__(self, *args, **kwargs)
        self._get_obs = self._override_get_obs
        obs = self._get_obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()
        original_obs = FetchPushEnv._get_obs(self)

        done = False
        info = {
            'is_success': self._is_success(original_obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(original_obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def _override_get_obs(self):
        obs = FetchPushEnv._get_obs(self)
        return np.concatenate([obs['observation'], obs['achieved_goal'], obs['desired_goal']])


class NongoalFetchPushDenseEnv(NongoalFetchPushEnv):

    def __init__(self, *args, **kwargs):
        NongoalFetchPushEnv.__init__(self, reward_type='dense', *args, **kwargs)



