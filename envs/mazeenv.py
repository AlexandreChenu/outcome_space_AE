import sys
import os
sys.path.append(os.getcwd())
from gym_marblemaze.envs.mazeenv.maze.maze import Maze
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

class State(object):
    def __init__(self, lst=None):
        if lst is not None:
            self.x = lst[0]
            self.y = lst[1]
        else:
            self.x = 0.5
            self.y = 0.5

    def to_list(self):
        return np.array([self.x, self.y])

    def distance_to(self,other):
        dx = self.x-other.x
        dy = self.y-other.y
        return math.sqrt(dx*dx+dy*dy)

    def act(self,action):
        a = action[:]
        if a[0]>.1:
            a[0]=.1
        if a[0]<-.1:
            a[0]=-.1
        if a[1]>.1:
            a[1]=.1
        if a[1]<-.1:
            a[1]=-.1
        r = State()
        r.x = self.x + a[0]
        r.y = self.y + a[1]
        return r

    def perturbation(self,mag=1e-5):
        r = State()
        r.x = self.x + mag * random.uniform(0,1)
        r.y = self.y + mag * random.uniform(0,1)
        return r

    def isInBounds(self,maze):
        return (self.x>0 and self.y>0 and self.x<maze.num_cols and self.y<maze.num_rows)

    def __str__(self):
        return "({:10.2f},{:10.2f})".format(self.x,self.y)

class MazeEnv(Maze, gym.Env):
    def __init__(self,args={
            'mazesize':15,
            'random_seed':0,
            'mazestandard':False,
            'wallthickness':0.1,
            'wallskill':True,
            'targetkills':True
        }):
        self.setup(args)
        self.allsteps = []
        self.lines = None
        self.interacts = 0

    def setup(self,args):
        super(MazeEnv,self).__init__(args['mazesize'],args['mazesize'],seed=args['random_seed'],standard=args['mazestandard'])
        ms = int(args['mazesize'])
        self.state = np.array([0.5, 0.5])
        self.steps = []
        self.obs_dim = 2
        self.thick = args['wallthickness']
        self.action_space = spaces.Box(np.array([-1,-1]),np.array([1,1]))
        self.observation_space = spaces.Box(np.array([0,0]),np.array([ms,ms]))

        self.metadata = dict()
        self.reward_range = [0.0, 2.0]
        #self.unwrapped = True
        self._configured = False

        self.alive_bonus=0.001
        self.distance_bonus=0.01
        self.obstacle_reward=0
        # self.obstacle_reward = 0
        self.target_reward=1
        #self.wallskill=args['wallskill']
        #self.targetkills=args['targetkills']
        self.wallskill = False
        self.targetkills = False


        self.goal = np.array([2.5, 0.5])
        self.width = 0.1


    def reward_function(self, obs, goal):
        d =  np.linalg.norm(obs[:2] - goal, axis=-1)

        if d > 0.10:
            return 1./d
        else:
            return 10.

    # def state_act(self, action):
    #     a = action[:]
    #
    #     if a[0]>1.:
    #         a[0]=1.
    #     if a[0]<-1.:
    #         a[0]=-1.
    #     if a[1]>1.:
    #         a[1]=1.
    #     if a[1]<-1.:
    #         a[1]=-1.
    #
    #     a = action[:]/10.
    #
    #     r = np.copy(self.state)
    #
    #     r[0] = r[0] + a[0]
    #     r[1] = r[1] + a[1]
    #
    #     return r

    def state_act(self, action):
        a = action[:]

        r = np.copy(self.state)

        r[0] = r[0] + a[0]
        r[1] = r[1] + a[1]

        return r

    def state_perturbation(self, mag=1e-5):
        r = np.copy(self.state)
        r[0] = r[0] + mag * random.uniform(0,1)
        r[1] = r[1] + mag * random.uniform(0,1)
        return r

    def state_isInBounds(self, state, maze):

        return (state[0]>0 and state[1]>0 and state[0] < maze.num_cols and state[1] < maze.num_rows)

    def seed(self,v):
        pass

    def reset(self):
        return self.reset_primitive()

    def reset_primitive(self):
        self.state = np.array([0.5, 0.5])
        self.steps = []
        return self.state

    def set_state(self,state):
        self.state = np.array([0.5, 0.5])

    def close(self):
        pass

    @property
    def dt(self):
        return 0.01

    def state_vector(self):
        return self.state

    def __seg_to_bb(self,s):
        bb = [list(s[0]),list(s[1])]
        if bb[0][0] > bb[1][0]:
            bb[0][0] = s[1][0]
            bb[1][0] = s[0][0]
        if bb[0][1] > bb[1][1]:
            bb[0][1] = s[1][1]
            bb[1][1] = s[0][1]
        return bb

    def __bb_intersect(self,a,b,e=1e-8):
        return (a[0][0] <= b[1][0] + e
                and a[1][0] + e >= b[0][0]
                and a[0][1] <= b[1][1] + e
                and a[1][1] + e >= b[0][1]);

    def __cross(self,a,b):
        return a[0]*b[1] - a[1]*b[0]

    def __is_point_right_of_line(self,a,b):
        atmp=[a[1][0] - a[0][0], a[1][1] - a[0][1]];
        btmp=[b[0] - a[0][0], b[1] - a[0][1]];
        return self.__cross(atmp,btmp) < 0;

    def __is_point_on_line(self,a,b,e=1e-8):
        atmp=[a[1][0] - a[0][0], a[1][1] - a[0][1]];
        btmp=[b[0] - a[0][0], b[1] - a[0][1]];
        return self.__cross(atmp,btmp) <= 1e-8;

    def __segment_touches_or_crosses_line(self,a,b,e=1e-8):
        return (self.__is_point_on_line(a,b[0],e)
            or self.__is_point_on_line(a,b[1],e)
            or (self.__is_point_right_of_line(a,b[0])
                != self.__is_point_right_of_line(a,b[1])))

    def __segments_intersect(self,a,b,e=1e-8):
        return (self.__bb_intersect(self.__seg_to_bb(a),self.__seg_to_bb(b),e)
            and self.__segment_touches_or_crosses_line(a,b,e)
            and self.__segment_touches_or_crosses_line(b,a,e))

    def state_in_wall(self,s):
        t = self.thick
        if t==0:
            return False
        def in_hwall(i,j,t=0):
            return s[0]>=i-t and s[0]<=i+t and s[1]>=j and s[1]<=j+1
        def in_vwall(i,j,t=0):
            return s[0]>=i and s[0]<i+1 and s[1]>=j-t and s[1]<=j+t
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.grid[i][j].walls["top"]:
                    if in_hwall(i,j,t):
                        return True
                if self.grid[i][j].walls["bottom"]:
                    if in_hwall(i+1,j,t):
                        return True
                if self.grid[i][j].walls["left"]:
                    if in_vwall(i,j,t):
                        return True
                if self.grid[i][j].walls["right"]:
                    if in_vwall(i,j+1,t):
                        return True
        return False

    def random_state(self):
        while True:
            s = self.num_cols * np.random.rand(2)
            if not self.state_in_wall(s):
                break
        return s

    def valid_action(self,action,cur_state=None):
        if cur_state is None:
            cur_state = self.state
        if len(action)==2:
            #if not cur_state.act(action).isInBounds(self):
            if not self.state_isInBounds(self.state_act(action), self):
                return False
            sa = self.state_act(action)

            #s = [(cur_state.x,cur_state.y),(sa.x,sa.y) ]
            s = [(cur_state[0],cur_state[1]),(sa[0],sa[1]) ]
            if self.lines is None:
                self.lines = [] #todo optim
                def add_hwall(lines,i,j,t=0):
                    lines.append([(i-t,j),(i-t,j+1)])
                    if t>0:
                        lines.append([(i-t,j+1),(i+t,j+1)])
                        lines.append([(i+t,j),(i+t,j+1)])
                        lines.append([(i+t,j),(i-t,j)])
                def add_vwall(lines,i,j,t=0):
                    lines.append([(i,j-t),(i+1,j-t)])
                    if t>0:
                        lines.append([(i+1,j-t),(i+1,j+t)])
                        lines.append([(i,j+t),(i+1,j+t)])
                        lines.append([(i,j+t),(i,j-t)])
                t = self.thick
                for i in range(len(self.grid)):
                    for j in range(len(self.grid[i])):
                        if self.grid[i][j].walls["top"]:
                            add_hwall(self.lines,i,j,t)
                        if self.grid[i][j].walls["bottom"]:
                            add_hwall(self.lines,i+1,j,t)
                        if self.grid[i][j].walls["left"]:
                            add_vwall(self.lines,i,j,t)
                        if self.grid[i][j].walls["right"]:
                            add_vwall(self.lines,i,j+1,t)
            for l in self.lines:
                if self.__segments_intersect(l,s):
                    return False
            return True
        else:
            return False

    def step(self,action):
        return self._step(action)

    def setConfig(self,args):
        self.alive_bonus = float(args['alivebonus'])
        self.distance_bonus = float(args['distancebonus'])
        self.obstacle_reward=float(args['obstaclereward'])
        self.target_reward=float(args['targetreward'])

    # "Static" function to simulate a step. Returns s', r, t
    def quickstep(self,state,action):
        if np.array(action).shape==(1,2):
            action = action[0]
        if abs(action[0])<1e-5 and abs(action[1])<1e-5:
            action[0] = 1e-5 * random.uniform(-1,1)
            action[1] = 1e-5 * random.uniform(-1,1)
        if not self.valid_action(action,cur_state=State(state)):
            #print("Invalid action ",action," with state ",self.state)
            #DEBUG : invalid actions are simply ignored
            #raise NotImplementedError #todo
            sp = self.state.perturbation(1e-9)
            return sp.to_list(),self.obstacle_reward,self.wallskill
        sp = State(state).act(action)
        if self.num_cols==2:
            dst = State(state).distance_to(State(lst=[1-.5,self.num_cols-.5]))
        else:
            dst = State(state).distance_to(State(lst=[self.num_rows-.5,self.num_cols-.5]))
        alive_bonus = self.alive_bonus
        target_bonus = self.target_reward if dst<.2 else 0.
        distance_bonus = 0. if self.distance_bonus==0 else self.distance_bonus/(dst+1)
        reward = alive_bonus + target_bonus + distance_bonus
        done = True if dst<.2 and self.targetkills else False
        return sp.to_list(),reward,done

    def _step(self,action):
        if np.array(action).shape==(1,2):
            action = action[0]
        if abs(action[0])<1e-5 and abs(action[1])<1e-5:
            action[0] = 1e-5 * random.uniform(-1,1)
            action[1] = 1e-5 * random.uniform(-1,1)
        if not self.valid_action(action):
            #DEBUG : invalid actions are simply ignored
            #raise NotImplementedError #todo
            #self.state = self.state_perturbation(1e-9)
            reward = self.reward_function(self.state, self.goal)
            #return self.state.to_list(), self.obstacle_reward, self.wallskill, {'target_reached': False}
            return list(self.state), reward + self.obstacle_reward, self.wallskill, {'target_reached': False}

        if self.valid_action(action):
            state_before=self.state
            #self.state = self.state.act(action)
            self.state = self.state_act(action)
            self.interacts += 1
            self.steps.append([state_before, action, self.state])

        #allsteps is obsolete ?
        self.allsteps.append([state_before, action, self.state])
        if len(self.allsteps)>10000:
            self.allsteps = self.allsteps[1:]

        # if self.num_cols==2:
        #     dst = self.state.distance_to(State(lst=[1-.5,self.num_cols-.5]))
        # else:
        #     dst = self.state.distance_to(State(lst=[self.num_rows-.5,self.num_cols-.5]))

        # alive_bonus = self.alive_bonus
        # target_bonus = self.target_reward if dst<.2 else 0.
        # distance_bonus = 0. if self.distance_bonus==0 else self.distance_bonus/(dst+1)
        #reward = alive_bonus + target_bonus + distance_bonus

        reward = self.reward_function(self.state, self.goal)

        # if np.linalg.norm(self.state, self.goal) < self.width:
        #     done = True
        #
        # else:
        #     done = False

        #done = True if dst<.2 and self.targetkills else False
        done = False # env never terminates

        #info = {'target_reached': dst<.2}
        info = {}

        #return self.state.to_list(),reward,done,info
        return list(self.state),reward,done,info


    def draw(self,ax,color=None,**kwargs):
        Maze.draw(self,ax,thick=self.thick)
        if self.num_cols==2:
            target = [1-.5,self.num_cols-.5]
        else:
            target = [self.num_rows-.5,self.num_cols-.5]
        #c = Circle(target,.2,fill=False)
        #ax.add_patch(c)
        if 'paths' in kwargs and kwargs['paths']:
            print("Drawing ",len(self.allsteps)," steps")
            lines = []
            colors = []
            s = len(self.allsteps)
            i = 0
            for b,ac,a in self.allsteps:
                lines.append([(b.x,b.y),(a.x,a.y)])
                if color is None:
                    colors.append([float(i)/s,0,0])
                else:
                    colors.append(color)
                i += 1
            lc = mc.LineCollection(lines, linewidths=1, colors=colors)
            ax.add_collection(lc)
            print("  ... done")


class FreeMazeEnv(MazeEnv, gym.Env):
    def __init__(self,args={
            'mazesize':15,
            'random_seed':0,
            'mazestandard':False,
            'wallthickness':0.1,
            'wallskill':True,
            'targetkills':True
        }):
        super(FreeMazeEnv,self).__init__(args = args)
        self.empty_grid()
