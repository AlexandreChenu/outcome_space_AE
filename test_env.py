import sys
import os
sys.path.append(os.getcwd())
#from gym_marblemaze.envs.mazeenv.maze.maze import Maze
from envs.mazeenv.mazeenv import *
#from maze.maze import Maze
from matplotlib import collections  as mc
from matplotlib.patches import Circle
import gym
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



def generate_demo(env, dir_path):
    """
    generate demonstrations
    env: environment
    dir_path: path to directory for demo saving
    """
    demos = []
    nb_demos = 500
    min_demo_len = 25
    max_demo_len = 50

    for i in range(0,nb_demos):
        starting_state = [np.random.rand() , np.random.rand()]
        #starting_state = [1.5,1.]
        budget = np.random.randint(min_demo_len, max_demo_len)

        env.reset()
        env.state = starting_state
        demo = []
        action = [0.,0.]
        for t in range(0,budget):
            new_action = env.action_space.sample()
            action = [0.7*action[0] + 0.3*new_action[0]/3., 0.7*action[1] + 0.3*new_action[1]/3.]
            new_state, _,_,_ = env.step(action)
            demo.append(new_state)

        demos.append(demo)

    print("len(demos) = ", len(demos))


    ## convert demos
    demos_masks = []
    demo_t = torch.zeros((nb_demos, max_demo_len,2))

    for indx_demo in range(0,len(demos)):
        # print("indx_demo = ", indx_demo)

        mask = []
        for indx_state in range(0,len(demos[indx_demo])):
            # print("indx_state = ", indx_state)
            # demo tensor
            demo_t[indx_demo][indx_state][0] = demos[indx_demo][indx_state][0]
            demo_t[indx_demo][indx_state][1] = demos[indx_demo][indx_state][1]

            # mask tensor
            mask.append(False)

        if len(demos[indx_demo]) < max_demo_len:
            for i in range(len(demos[indx_demo]), max_demo_len):
                mask.append(True)

        demos_masks.append(mask)

    print("len(demos_masks) = ", len(demos_masks))
    demos_masks_t = torch.as_tensor(demos_masks, dtype=torch.bool)

    with open(dir_path + '/demo_observations.pickle', 'wb') as handle:
        pickle.dump(demo_t, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(dir_path + '/demo_masks.pickle', 'wb') as handle:
        pickle.dump(demos_masks_t, handle, protocol=pickle.HIGHEST_PROTOCOL)

def sample_trajectories(env, population):
    trajectories = []

    for individual in population:
        trajectory = []
        ob = env.reset()
        while env.done == False:
            action = individual(ob)
            new_ob,_,_,_ = env.step(action)
            trajectory.append(ob)
            ob = new_ob
        trajectories.append(trajectory)

    return trajectories

def random_agent(obs):
    return np.random.uniform(-1.,1.,2)



if (__name__=='__main__'):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")

    cur_path = os.getcwd()
    dir_path = cur_path + "/demos/demos_" + dt_string + "_" + str(np.random.randint(1000))

    # use different name is directory already exists
    new_path = dir_path
    i = 0
    while path.exists(new_path):
        new_path = dir_path + "_" + str(i)
        i += 1

    dir_path = new_path

    try:
        os.mkdir(dir_path)
    except OSError:
        print ("Creation of the directory %s failed" % dir_path)
    else:
        print ("Successfully created the directory %s " % dir_path)


    env = gym.make("DubinsMazeEnv-v0", args= {
            'mazesize':2,
            'random_seed':0,
            'mazestandard':False,
            'wallthickness':0.1,
            'wallskill':True,
            'targetkills':True})
    env.empty_grid() ## remove walls for now

    print("env = ", env)

    population = [random_agent]

    trajs = sample_trajectories(env, population)

    print("trajs = ", trajs)
