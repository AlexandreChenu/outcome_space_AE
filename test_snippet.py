import sys
import os
sys.path.append(os.getcwd())
#from gym_marblemaze.envs.mazeenv.maze.maze import Maze
from envs.mazeenv.mazeenv import *
#from maze.maze import Maze
from matplotlib import collections  as mc
from matplotlib.patches import Circle
import gym
from gym.version import VERSION
print(VERSION)
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import math
import random
import copy
from collections import OrderedDict
import torch
import pickle
from datetime import datetime

import datetime
import os
from os import path
import array
import time
import random
import pickle
import copy
import argparse

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import collections as mc
from mpl_toolkits.mplot3d import Axes3D


def plot_demo(env, demo):

    fig, ax = plt.subplots()
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.1, 2.1)
    env.draw(ax)

    L_X = [state[0] for state in demo]
    L_Y = [state[1] for state in demo]

    plt.plot(L_X, L_Y, c="pink")
    plt.show()

    return

def sample_trajectories(env, population):
    trajectories = []

    for individual in population:
        trajectory = []
        ob = env.reset()
        while env.done == False:
            action = individual(ob)
            new_ob, reward, done, info = env.step(action)
            trajectory.append(ob)
            ob = new_ob
        trajectories.append(trajectory)

    return trajectories

def random_agent(obs):
    return np.random.uniform(-1.,1.,2)



if (__name__=='__main__'):

    ### create environment
    env = gym.make("DubinsMazeEnv-v0", args= {
            'mazesize':2,
            'random_seed':0,
            'mazestandard':False,
            'wallthickness':0.1,
            'wallskill':True,
            'targetkills':True})
    ## remove walls for now
    env.empty_grid()

    print("env = ", env)

    ## fake population
    population = [random_agent]

    trajs = sample_trajectories(env, population)

    plot_demo(env, trajs[0])
