import sys
import os
sys.path.append(os.getcwd())
#from gym_marblemaze.envs.mazeenv.maze.maze import Maze
from gym_marblemaze.envs.dubins_mazeenv.mazeenv import *
# from gym_marblemaze.envs.mazeenv.mazeenv import *
#from maze.maze import Maze
from matplotlib import collections  as mc
from matplotlib.patches import Circle
import gym
import gym_marblemaze
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import math
import random
import copy
from collections import OrderedDict
import torch


print("REGISTERING DubinMazeEnv_AE-v0")
register(
    id='DubinsMazeEnv_AE-v0',
    entry_point='gym_marblemaze.envs:DuinsMazeEnv_AE',
    kwargs={'args': {
            'mazesize':5,
            'random_seed':0,
            'mazestandard':False,
            'wallthickness':0.1,
            'wallskill':True,
            'targetkills':True,
            'max_steps':50
        }}
)


class DubinsMazeEnv_AE(DubinsMazeEnv):

    def __init__(self, args={
            'mazesize':15,
            'random_seed':0,
            'mazestandard':False,
            'wallthickness':0.1,
            'wallskill':True,
            'targetkills':True,
            'max_steps': 50
        }):

        #MazeEnv.__init__(self, args = args)
        super(DubinsMazeEnv_BP_SB3,self).__init__(args = args)

        print("MazeEnv.state = ", self.state)

        self.max_steps = args['max_steps']
        # Counter of steps per episode
        self.args = args
        self.rollout_steps = 0

        self.action_space = spaces.Box(np.array([0., -1.]), np.array([1., 1.]), dtype = np.float32)
        low = np.array([0.,0.,-4.])
        high = np.array([args['mazesize'], args['mazesize'],4.])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Counter of steps per episode
        self.rollout_steps = 0
        self.demo_traj = demo_traj
        self.starting_indx = 0
        self.width = 0.5
        self.traj = []

    def state_act(self, action):
        r = np.copy(self.state)
        return self.update_state(r, action, self.delta_t)

    def step(self, action) :

        obs = self.state_vector()
        new_obs, reward, done, info = self._step(action)
        self.traj.append(new_obs)

        self.rollout_steps += 1

        dst = np.linalg.norm(new_obs[:2] - self.goal[:2])
        info = {'target_reached': dst<= self.width}

        if info['target_reached']:
            done = True

        if info['target_reached']: # achieved goal
            if done or self.rollout_steps >= self.max_steps:
                done = True
                info['done'] = done
                info['traj'] = self.traj
                self.reset()
                return new_obs, 1., done, info

        else:
            if self.rollout_steps >= self.max_steps:
                done = True
                info['done'] = done
                info['traj'] = self.traj
                self.reset()
                return new_obs, 0., done, info
            else:
                info['done'] = done
                return new_obs, 0., done, info

    def set_state(self, state):
        state = self.observation_space.sample()
        self.state = state
        return self.state_vector()

    def set_task_length(self, task_length):
        self.max_steps = task_length
        return 0

    def set_goal_state(self, goal_state):
        self.goal = goal_state
        return 0

    def reset(self):
        # print("state = ", self.state_vector())
        self.reset_primitive()
        self.set_state()
        self.task_length = len(self.demo_traj) - self.starting_indx
        self.rollout_steps = 0
        self.traj = []
        self.traj.append(self.state_vector())

        return self.state_vector()
