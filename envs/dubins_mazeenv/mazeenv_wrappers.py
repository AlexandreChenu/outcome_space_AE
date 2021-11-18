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

from .task_manager_mazeenv import *

class DubinsMazeEnvGCPSB3(DubinsMazeEnv):

    def __init__(self, L_states, L_steps, args={
            'mazesize':5,
            'random_seed':0,
            'mazestandard':False,
            'wallthickness':0.1,
            'wallskill':True,
            'targetkills':True,
            'max_steps': 50,
            'width': 0.1
        }):

        #MazeEnv.__init__(self, args = args)
        super(DubinsMazeEnvGCPSB3,self).__init__(args = args)

        print("state = ", self.state)

        self.max_steps = args['max_steps']
        # Counter of steps per episode
        self.args = args
        self.rollout_steps = 0

        self.action_space = spaces.Box(np.array([0., -1.]), np.array([1., 1.]), dtype = np.float32)
        low = np.array([0.,0.,-4., 0., 0.])
        high = np.array([args['mazesize'], args['mazesize'],4., args['mazesize'], args['mazesize']])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.state =  np.array([0.5, 0.5, 0.])
        self.goal= np.array([args['mazesize']-0.5, args['mazesize']-0.5])
        self.width = args['width']
        self.width = 0.3
        self.width_test = 0.3
        self.L_states = L_states
        self.L_steps = L_steps
        self.success_bonus = 10
        self.traj = []
        self.indx_start = 0
        self.indx_goal = -1
        self.testing = False
        self.expanded = False

        self.buffer_transitions = []
        self.go_explore_prob = 0.

    def state_act(self, action):
        r = np.copy(self.state)
        return self.update_state(r, action, self.delta_t)


    def step(self, action) :

        obs = self.state_vector()
        new_obs, reward, done, info = self._step(action)
        self.traj.append(new_obs[:2])

        self.rollout_steps += 1

        dst = np.linalg.norm(self.state[:2] - self.goal)
        info = {'target_reached': dst<= self.width}
        reward = -dst

        if info['target_reached']:
            done = True

        if info['target_reached']: # achieved goal

            if not self.testing and not self.expanded and random.random() < self.go_explore_prob and self.indx_goal < len(self.L_states) - 1:
                ## continue learning with next goal if not testing and next goal available
                print("Go_explore trick")

                self.expanded = True
                info['done'] = True
                info['traj'] = self.traj
                # done = True
                #print("self.goal = ", type(self.goal))
                prev_goal = copy.deepcopy(self.goal) ## save previous goal for transitions
                #print("prev_goal = ", type(prev_goal))

                ## modify goal and extend trajectory for next goal -> no reset()
                self.indx_start += 1
                self.indx_goal += 1
                length_task = sum(self.L_steps[self.indx_start:self.indx_goal]) ## bonus timesteps
                goal_state = self.L_states[self.indx_goal]
                self.set_goal_state(goal_state)
                self.max_steps = length_task
                self.rollout_steps = 0

                new_dst = np.linalg.norm(self.state[:2] - self.goal)
                new_info = {'target_reached': new_dst<= self.width}
                new_info['done'] = False
                new_done = False
                new_reward = -new_dst

                self.buffer_transitions.append((action, np.hstack((new_obs, self.goal)), new_reward, new_done, new_info))

                return np.hstack((new_obs, prev_goal)), reward, done, info
                #return np.hstack((new_obs, self.goal)), reward, done, info

            else:
                prev_goal = copy.deepcopy(self.goal)
                info['done'] = done
                info['traj'] = self.traj
                self.reset()
                return np.hstack((new_obs, prev_goal)), reward, done, info

        elif self.rollout_steps >= self.max_steps:
            done = True
            prev_goal = copy.deepcopy(self.goal)
            info['done'] = done
            info['traj'] = self.traj
            self.reset()
            return np.hstack((new_obs, prev_goal)), reward, done, info

        else:
            info['done'] = done
            #info['traj'] = self.traj
            return np.hstack((new_obs, self.goal)), reward, done, info

    def step_test(self, action) :

        obs = self.state_vector()
        new_obs, reward, done, info = self._step(action)
        self.traj.append(new_obs[:2])

        self.rollout_steps += 1

        dst = np.linalg.norm(self.state[:2] - self.goal)
        info = {'target_reached': dst<= self.width_test}
        reward = -dst

        return np.hstack((new_obs, self.goal)), reward, done, info

    def state_vector(self):
        return self.state

    def goal_vector(self):
        return self.goal

    def set_state(self, starting_state):
        self.state = np.array(starting_state)
        return self.state_vector()

    def set_goal_state(self, goal_state):
        self.goal = np.array(goal_state)[:2]
        return 0

    def sample_task(self):
        """
        Sample task for low-level policy training
        """

        delta_step = 1
        self.indx_start = random.randint(0, len(self.L_states) - delta_step -1)
        self.indx_goal = self.indx_start + delta_step
        length_task = sum(self.L_steps[self.indx_start:self.indx_goal])
        starting_state = self.L_states[self.indx_start]
        goal_state = self.L_states[self.indx_goal]

        return starting_state, length_task, goal_state

    def reset(self):
        # print("state = ", self.state_vector())
        self.reset_primitive()
        self.testing = False

        starting_state, length_task, goal_state = self.sample_task()

        self.set_goal_state(goal_state)
        self.set_state(starting_state)
        self.max_steps = length_task

        self.rollout_steps = 0
        self.traj = []
        self.traj.append(self.state_vector()[:2])

        #print("reset = ", np.hstack((self.state_vector(), self.goal)))

        return np.hstack((self.state_vector(), self.goal))





class DubinsMazeEnvGCPHERSB3(DubinsMazeEnv):

    def __init__(self, L_states, L_steps, args={
            'mazesize':5,
            'random_seed':0,
            'mazestandard':False,
            'wallthickness':0.1,
            'wallskill':True,
            'targetkills':True,
            'max_steps': 50,
            'width': 0.1
        }):

        super(DubinsMazeEnvGCPHERSB3,self).__init__(args = args)

        print("state = ", self.state)

        self.max_steps = args['max_steps']
        # Counter of steps per episode
        self.args = args
        self.rollout_steps = 0

        self.action_space = spaces.Box(np.array([0., -1.]), np.array([1., 1.]), dtype = np.float32)


        ms = int(args['mazesize'])
        self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(
                        low = np.array([0,0,-4]),
                        high = np.array([ms,ms,4])),
                    "achieved_goal": spaces.Box(
                        low = np.array([0,0]),
                        high = np.array([ms,ms])),
                    "desired_goal": spaces.Box(
                        low = np.array([0,0]),
                        high = np.array([ms,ms])),
                }
            )

        self.state =  np.array([0.5, 0.5, 0.])
        self.goal= np.array([args['mazesize']-0.5, args['mazesize']-0.5])
        # self.width = args['width']
        self.width_bonus = 0.3
        self.width_success = 0.3
        self.width = self.width_bonus
        self.width_test = self.width_success
        self.L_states = L_states
        self.L_steps = L_steps
        self.total_steps = 0
        self.starting_state_set = []
        self.L_runs_results = []
        self.results_size = 10

        self.success_bonus = 10
        self.traj = []
        self.indx_start = 0
        self.indx_goal = -1
        self.testing = False
        self.expanded = False

        self.buffer_transitions = []

        self.bonus = True
        self.weighted_selection = True

        self.target_selection = False
        self.target_ratio = 0.3
        self.model = None

        self.skipping_ratio = 0.25
        self.skipping = False
        self.allow_skipping = False
        self.skipping_results = []

        self.skipping_success_ratio = 0.75

    def state_act(self, action):
        r = np.copy(self.state)
        return self.update_state(r, action, self.delta_t)

    # def compute_reward(self, achieved_goal, desired_goal, info):
    #
    #
    #     if len(achieved_goal.shape) ==  1:
    #         dst = np.linalg.norm(self.goal_space_projection(achieved_goal) - self.goal_space_projection(desired_goal), axis = -1)
    #         _info = {'target_reached': dst<= self.width}
    #
    #         assert _info['target_reached'] == info['target_reached']
    #
    #         if _info['target_reached']:
    #
    #             if self.bonus and info['goal_prime_available']:
    #
    #                 L = info['task_length']
    #                 #L = max(self.L_steps) + 1
    #                 goal_prime = info['goal_prime']
    #
    #                 obs = OrderedDict([("observation", achieved_goal.copy()),
    #                         ("achieved_goal", achieved_goal.copy()),
    #                         ("desired_goal",  achieved_goal.copy())])
    #                 t_obs = obs.copy()
    #                 t_obs["observation"] = torch.FloatTensor([t_obs["observation"]])
    #                 ## change desired goal for next one
    #                 t_obs["desired_goal"] = torch.FloatTensor([goal_prime.copy()])
    #                 t_obs["achieved_goal"] = torch.FloatTensor([t_obs["achieved_goal"]])
    #
    #                 ## compute value
    #                 action, _ = self.model.policy.actor.predict(t_obs, deterministic=True)
    #                 action = torch.FloatTensor([action])[0]
    #                 value = self.model.critic(t_obs, action)
    #
    #
    #                 value = float(value[0])
    #
    #                 if self.model.num_timesteps > 50000: ## critic shouldn't be too wrong
    #                 #if value <= 0.: ### should make sure that the value converged more or less (at least, that the entropy term is low)
    #                     # value -= 0.00001 ## avoid dividing by zero
    #                     #
    #                     # inv_value = 1./abs(value) # [0. , inf]
    #                     # clipped_inv_value = min(inv_value, 10.) # [0., 10.]
    #                     # bonus_reward = 0.01*clipped_inv_value # [0., 0.1]
    #
    #                     diff_value = L + value # [0., 6.]
    #                     bonus_reward = diff_value*0.01  # [0., 0.6]
    #
    #                 else:
    #                     bonus_reward = 0.
    #
    #
    #                 return -0.1 + bonus_reward
    #             else:
    #                 return -0.1
    #         else:
    #             return -1.
    #
    #     else:
    #
    #         # dst = np.linalg.norm(self.goal_space_projection(achieved_goal) - self.goal_space_projection(desired_goal), axis = -1)
    #         # _info = {'target_reached': dst<= self.width}
    #
    #         # for _info in info:
    #         #     if _info['target_reached']:
    #         #         print("info = ", _info)
    #         #print("info = ", info)
    #
    #         obs = OrderedDict([("observation", achieved_goal.copy()),
    #                 ("achieved_goal", achieved_goal.copy()),
    #                 ("desired_goal", desired_goal.copy())])
    #
    #         t_obs = obs.copy()
    #         t_obs["observation"] = torch.FloatTensor(t_obs["observation"])
    #         t_obs["desired_goal"] = torch.FloatTensor(t_obs["desired_goal"])
    #         t_obs["achieved_goal"] = torch.FloatTensor(t_obs["achieved_goal"])
    #
    #         action, _ = self.model.policy.actor.predict(t_obs, deterministic=True)
    #         action = torch.FloatTensor([action])[0]
    #
    #         value = self.model.critic(t_obs, action)
    #
    #         # print("achieved_goal = ", achieved_goal)
    #         # print("achieved_goal[:,:2] = ", achieved_goal[:,:2])
    #
    #         distances = np.linalg.norm(achieved_goal[:,:2] - desired_goal[:,:2], axis=-1)  ## pour éviter le cas ou dist == 0.
    #
    #         distances_mask = (distances <= self.width).astype(np.float32)
    #
    #         # rewards = distances_mask - 1. # {-1, 0.}
    #         rewards = distances_mask - 1. - distances_mask * 0.1 # {-1, -0.1}
    #
    #         return rewards


    def compute_reward(self, achieved_goal, desired_goal, info):

        if len(achieved_goal.shape) ==  1:
            dst = np.linalg.norm(self.goal_space_projection(achieved_goal) - self.goal_space_projection(desired_goal), axis = -1)
            _info = {'target_reached': dst<= self.width_success}

            if dst <= self.width_bonus:
                _info['bonus'] = True
            else:
                _info['bonus'] = False

            assert _info['target_reached'] == info['target_reached']

            if _info['bonus']:

                #bonus_reached = 0.1 - (dst / self.width_bonus) * 0.1
                bonus_reached = 0.

                if self.bonus and info['goal_prime_available']:

                    L = info['next_task_length']
                    #L = max(self.L_steps) + 1
                    goal_prime = info['goal_prime']
                    goal_prime = self.goal_space_projection(goal_prime)
                    #assert goal_prime.shape[0] == self.observation_space["desired_goal"].shape[0]

                    obs = OrderedDict([("observation", achieved_goal.copy()),
                            ("achieved_goal", self.goal_space_projection(achieved_goal.copy())),
                            ("desired_goal",  self.goal_space_projection(achieved_goal.copy()))])

                    t_obs = obs.copy()

                    t_obs["observation"] = torch.FloatTensor([t_obs["observation"]])
                    assert t_obs["observation"].shape[1] == self.observation_space["observation"].shape[0]

                    ## change desired goal for next one
                    t_obs["desired_goal"] = torch.FloatTensor([goal_prime.copy()])
                    assert t_obs["desired_goal"].shape[1] == self.observation_space["desired_goal"].shape[0]

                    t_obs["achieved_goal"] = torch.FloatTensor([t_obs["achieved_goal"]])
                    assert t_obs["achieved_goal"].shape[1] == self.observation_space["achieved_goal"].shape[0]

                    ## compute value
                    action, _ = self.model.policy.actor.predict(t_obs, deterministic=True)
                    action = torch.FloatTensor([action])[0]
                    value = self.model.critic(t_obs, action)

                    value = float(value[0])

                    # if self.model.num_timesteps > 35000: ## critic shouldn't be too wrong
                    # #if value <= 0.: ### should make sure that the value converged more or less (at least, that the entropy term is low)
                    #     # value -= 0.00001 ## avoid dividing by zero
                    #     #
                    #     # inv_value = 1./abs(value) # [0. , inf]
                    #     # clipped_inv_value = min(inv_value, 10.) # [0., 10.]
                    #     # bonus_reward = 0.01*clipped_inv_value # [0., 0.1]
                    #
                    #     if value < int(L/2):
                    #         reward_penalty = -0.5
                    #
                    #     else:
                    #         reward_penalty = 0.
                    #     #
                    #     # diff_value = L + value # [0., 6.]
                    #     # bonus_reward = diff_value*0.01  # [0., 0.6]
                    #
                    # else:
                    #     reward_penalty = 0.

                    ### penalty 2

                    ## can be called even if the Q function is not a good approximation of the trajectory length
                    # if value <  - L - int(L/2) :
                    #     reward_penalty = -0.1
                    #
                    # else:
                    #     reward_penalty = 0.


                    ### penalty 3
                    # print("indx_start = ", self.indx_start)
                    # print("L = ", L)
                    # print("value = ", value)
                    #
                    # if int(L/2) + value <  0. : ## make sure we converged
                    #     reward_penalty = (int(L/2) + value)*0.01
                    #
                    # else:
                    #     reward_penalty = 0.
                    #
                    # print("reward penalty = ", reward_penalty)

                    ## bonus 4 (also 1)
                    if self.model.num_timesteps > 45000:
                        diff_value = L + value
                        bonus_value = diff_value*0.01

                        if bonus_value >= 0.1:
                            print("\n LARGE BONUS VALUE:")
                            print("L = ", L)
                            print("value = ", value)

                    else:
                        bonus_value = 0.

                    # return -0.2 + bonus_value + bonus_reached
                    return -0.1 + bonus_value + bonus_reached

                else:
                    # return -0.2 + bonus_reached
                    return -0.1 + bonus_reached

            else:
                return -1.

        else:
            ## compute -1 or 0 reward
            distances = np.linalg.norm(achieved_goal[:,:2] - desired_goal[:,:2], axis=-1)  ## pour éviter le cas ou dist == 0.
            distances_mask = (distances <= self.width_bonus).astype(np.float32)

            # rewards = distances_mask - 1. # {-1, 0.}
            # rewards = distances_mask - 1. - distances_mask * 0.2 # {-1, -0.1}
            rewards = distances_mask - 1. - distances_mask * 0.1 # {-1, -0.1}

            return rewards

        # #### add bonus reward even after relabelling
        # ########################################################################
        # ########################################################################
        #     rewards = []
        #     for i in range(len(info)):
        #         R = 0.
        #         dst = np.linalg.norm(self.goal_space_projection(achieved_goal[i]) - self.goal_space_projection(desired_goal[i]))
        #         _info = {'target_reached': dst<= self.width_success}
        #
        #         if dst <= self.width_bonus:
        #             _info['bonus'] = True
        #         else:
        #             _info['bonus'] = False
        #
        #         if _info['bonus']:
        #             bonus_value = 0.
        #             if self.bonus and info[i]['goal_prime_available']:
        #                 adg = info[i]['actual_desired_goal']
        #                 if np.linalg.norm(self.goal_space_projection(adg) - self.goal_space_projection(desired_goal[i])) <= self.width_success:
        #                     ## desired_goal is close enough from the actual task goal for the bonus to be meaningful
        #
        #                     L = info[i]['next_task_length']
        #                     observation = info[i]['new_observation']
        #                     goal_prime = info[i]['goal_prime'] # next task goal
        #                     goal_prime = self.goal_space_projection(goal_prime)
        #
        #                     assert observation[0] == achieved_goal[i][0] and observation[1] == achieved_goal[i][1]
        #
        #                     obs = OrderedDict([("observation", observation.copy()),
        #                             ("achieved_goal",  self.goal_space_projection(achieved_goal[i].copy())),
        #                             ("desired_goal", self.goal_space_projection(goal_prime.copy()))])
        #
        #                     t_obs = obs.copy()
        #
        #                     t_obs["observation"] = torch.FloatTensor([t_obs["observation"]])
        #                     assert t_obs["observation"].shape[1] == self.observation_space["observation"].shape[0]
        #
        #                     ## change desired goal for next one
        #                     t_obs["desired_goal"] = torch.FloatTensor([t_obs["desired_goal"]])
        #                     assert t_obs["desired_goal"].shape[1] == self.observation_space["desired_goal"].shape[0]
        #
        #                     t_obs["achieved_goal"] = torch.FloatTensor([t_obs["achieved_goal"]])
        #                     assert t_obs["achieved_goal"].shape[1] == self.observation_space["achieved_goal"].shape[0]
        #
        #                     ## compute value
        #                     action, _ = self.model.policy.actor.predict(t_obs, deterministic=True)
        #                     action = torch.FloatTensor([action])[0]
        #                     value = self.model.critic(t_obs, action)
        #
        #                     value = float(value[0])
        #
        #                     if self.model.num_timesteps > 45000:
        #                         diff_value = L + value
        #                         bonus_value += diff_value*0.01
        #
        #             R = -0.1 + bonus_value
        #
        #         else:
        #             R = -1.
        #
        #         rewards.append(R)
        #     ########################################################################
        #     ########################################################################
        #
        #     return np.array(rewards)

    def step(self, action) :

        self.last_obs = OrderedDict([
                ("observation", self.state.copy()),
                ("achieved_goal", self.goal_space_projection(self.state).copy()),
                ("desired_goal", self.goal_space_projection(self.goal).copy()),])

        state = self.state_vector()
        new_state, reward, done, info = self._step(action)
        self.traj.append(new_state)

        self.rollout_steps += 1

        dst = np.linalg.norm(self.goal_space_projection(self.state) - self.goal_space_projection(self.goal))
        info = {'target_reached': dst<= self.width_success}

        if dst <= self.width_bonus:
            info['bonus']=True
        else:
            info['bonus']=False

        self.buffer_transitions = [] ## reset to empty list before, eventually adding transitions

        ### update information for reward computation
        info['goal_prime_available'] = False

        if self.indx_goal  < len(self.L_states) - 1:
            info['goal_prime_available'] = True
            ## adg
            info['actual_desired_goal'] = self.goal_space_projection(self.goal).copy()
            ## goal prime
            goal_prime = self.goal_space_projection(self.L_states[self.indx_goal+1]).copy()
            info['goal_prime'] = goal_prime
            ## task length
            info['next_task_length'] = sum(self.L_steps[self.indx_goal:self.indx_goal+1]) #+ 5
            ## new observation
            info['new_observation'] = new_state.copy()


        if info['target_reached']: # achieved goal

            self.L_runs_results[self.indx_start].append(1)

            if len(self.L_runs_results[self.indx_start]) > self.results_size:
                self.L_runs_results[self.indx_start].pop(0)

            assert len(self.L_runs_results[self.indx_start]) <= self.results_size

            done = True

            reward = self.compute_reward(self.state, self.goal, info)

            # if self.indx_start == 9:
            #     print("target_reached | reward = ", reward)
            #
            #     obs = OrderedDict([("observation", state.copy()),
            #             ("achieved_goal", self.goal_space_projection(state.copy())),
            #             ("desired_goal",  self.goal_space_projection(self.goal.copy()))])
            #     t_obs = obs.copy()
            #
            #     t_obs["observation"] = torch.FloatTensor([t_obs["observation"]])
            #     assert t_obs["observation"].shape[1] == self.observation_space["observation"].shape[0]
            #     t_obs["desired_goal"] = torch.FloatTensor([t_obs["desired_goal"]])
            #     assert t_obs["desired_goal"].shape[1] == self.observation_space["desired_goal"].shape[0]
            #     t_obs["achieved_goal"] = torch.FloatTensor([t_obs["achieved_goal"]])
            #     assert t_obs["achieved_goal"].shape[1] == self.observation_space["achieved_goal"].shape[0]
            #
            #     ## compute value
            #     t_action = torch.FloatTensor([action])
            #     value = self.model.critic(t_obs, t_action)
            #     value = float(value[0])
            #
            #     print("target_reached | value = ", value)

            if self.indx_goal < len(self.starting_state_set) :
                ## add new sample to the set of starting_states
                self.starting_state_set[self.indx_goal].append(new_state)

            ## process skipping
            # if self.allow_skipping and self.skipping == True:
            #     self.skipping_cnts[self.indx_goal-1] += 1
            #
            #     if self.skipping_cnts[self.indx_goal-1] > 50:
            #
            #         # remove skipping counter
            #         self.skipping_cnts.pop(self.indx_goal-1)
            #
            #         # remove starting state
            #         self.starting_state_set.pop(self.indx_goal-1)
            #         self.L_states.pop(self.indx_goal-1)
            #
            #         # remove steps after adding them to the kept starting state
            #         self.L_steps[self.indx_start] += self.L_steps[self.indx_goal-1]
            #         self.L_steps.pop(self.indx_goal-1)
            #
            #         ## reset counters to 0 to avoid multiple removal in a row
            #         self.skipping_cnt = [0 for i in range(len(self.skipping_cnts))]
            #
            #     self.skipping = False


            ## UPDATE SKIPPING RESULTS
            if self.allow_skipping and self.skipping == True:
                self.skipping_results[self.indx_goal].append(1)

                if len(self.skipping_results[self.indx_goal]) > self.results_size:

                    self.skipping_results[self.indx_goal].pop(0)

                    nb_skip_success = self.skipping_results[self.indx_goal].count(1)
                    nb_skip_fails = self.skipping_results[self.indx_goal].count(0)

                    s_r = float(nb_skip_success/(nb_skip_fails + nb_skip_success))
                    assert s_r >= 0. and s_r <= 1.
                    print("success_ratio skip nb " + str(self.indx_goal) + " = " + str(s_r))

                    if s_r > self.skipping_success_ratio :
                        # remove starting state
                        self.starting_state_set.pop(self.indx_goal-1)
                        self.L_states.pop(self.indx_goal-1)

                        print("self.indx_start = ", self.indx_start)
                        print("self.indx_goal -1  = ", self.indx_goal -1)
                        print("self.L_steps[self.indx_start] = ", self.L_steps[self.indx_start])
                        print("self.L_steps[self.indx_goal-1] = ", self.L_steps[self.indx_goal-1])
                        print("self.L_steps = ", self.L_steps)

                        # remove steps after adding them to the kept starting state
                        self.L_steps[self.indx_start] += self.L_steps[self.indx_goal-1]
                        print("self.L_steps[self.indx_start] = ", self.L_steps[self.indx_start])

                        self.L_steps.pop(self.indx_goal-1)
                        print("self.L_steps = ", self.L_steps)

                        assert sum(self.L_steps) == self.total_steps # make sure we kept enough timesteps

                        # empty results buffer for the successfuly reached goal and the next goal
                        if self.indx_goal + 1 < len(self.skipping_results):
                            self.skipping_results[self.indx_goal+1] = []
                        self.skipping_results[self.indx_goal] = []

                        # remove skipping results for the skipped index
                        self.skipping_results.pop(self.indx_goal-1)

                        # make sure we removed everything properly
                        assert len(self.L_states) == len(self.L_steps) and len(self.L_steps) == len(self.skipping_results)


            prev_goal = copy.deepcopy(self.goal)
            info['done'] = done
            info['traj'] = self.traj
            self.reset()
            return OrderedDict([
                    ("observation", self.state.copy()),
                    ("achieved_goal", self.goal_space_projection(self.state).copy()),
                    ("desired_goal", self.goal_space_projection(prev_goal).copy()),]), reward, done, info

        elif self.rollout_steps >= self.max_steps:

            self.L_runs_results[self.indx_start].append(0)

            if len(self.L_runs_results[self.indx_start]) > self.results_size:
                self.L_runs_results[self.indx_start].pop(0)

            ## UPDATE SKIPPING RESULTS (NO SKIPPING CONFIRMATION HERE)
            if self.allow_skipping and self.skipping == True:
                print("self.indx_goal = ", self.indx_goal)
                print("len(self.skipping_results) = ", len(self.skipping_results))
                self.skipping_results[self.indx_goal].append(0)

                if len(self.skipping_results[self.indx_goal]) > self.results_size:
                    self.skipping_results[self.indx_goal].pop(0)

            reward = self.compute_reward(self.state, self.goal, info)

            # if self.indx_start == 9:
            #     print("timeout | reward = ", reward)
            #
            #     obs = OrderedDict([("observation", state.copy()),
            #             ("achieved_goal", self.goal_space_projection(state.copy())),
            #             ("desired_goal",  self.goal_space_projection(self.goal.copy()))])
            #     t_obs = obs.copy()
            #
            #     t_obs["observation"] = torch.FloatTensor([t_obs["observation"]])
            #     assert t_obs["observation"].shape[1] == self.observation_space["observation"].shape[0]
            #     t_obs["desired_goal"] = torch.FloatTensor([t_obs["desired_goal"]])
            #     assert t_obs["desired_goal"].shape[1] == self.observation_space["desired_goal"].shape[0]
            #     t_obs["achieved_goal"] = torch.FloatTensor([t_obs["achieved_goal"]])
            #     assert t_obs["achieved_goal"].shape[1] == self.observation_space["achieved_goal"].shape[0]
            #
            #     ## compute value
            #     t_action = torch.FloatTensor([action])
            #     value = self.model.critic(t_obs, t_action)
            #     value = float(value[0])
            #
            #     print("timeout | value = ", value)

            done = True
            prev_goal = copy.deepcopy(self.goal)
            info['done'] = done
            info['traj'] = self.traj
            self.reset()

            return OrderedDict([
                    ("observation", self.state.copy()),
                    ("achieved_goal", self.goal_space_projection(self.state).copy()),
                    ("desired_goal", self.goal_space_projection(prev_goal).copy()),]), reward, done, info

        else:

            done = False

            reward = self.compute_reward(self.state, self.goal, info)

            # if self.indx_start == 9:
            #     print("running | reward = ", reward)
            #
            #     obs = OrderedDict([("observation", state.copy()),
            #             ("achieved_goal", self.goal_space_projection(state.copy())),
            #             ("desired_goal",  self.goal_space_projection(self.goal.copy()))])
            #     t_obs = obs.copy()
            #
            #     t_obs["observation"] = torch.FloatTensor([t_obs["observation"]])
            #     assert t_obs["observation"].shape[1] == self.observation_space["observation"].shape[0]
            #     t_obs["desired_goal"] = torch.FloatTensor([t_obs["desired_goal"]])
            #     assert t_obs["desired_goal"].shape[1] == self.observation_space["desired_goal"].shape[0]
            #     t_obs["achieved_goal"] = torch.FloatTensor([t_obs["achieved_goal"]])
            #     assert t_obs["achieved_goal"].shape[1] == self.observation_space["achieved_goal"].shape[0]
            #
            #     #print("t_obs = ", t_obs)
            #     ## compute value
            #     t_action = torch.FloatTensor([action])
            #     value = self.model.critic(t_obs, t_action)
            #     value = float(value[0])
            #
            #     print("running | value = ", value)

            info['done'] = done
            return OrderedDict([
                    ("observation", self.state.copy()),
                    ("achieved_goal", self.goal_space_projection(self.state).copy()),
                    ("desired_goal", self.goal_space_projection(self.goal).copy()),]), reward, done, info

    def step_test(self, action) :

        obs = self.state_vector()
        new_obs, reward, done, info = self._step(action)
        self.traj.append(new_obs[:2])

        self.rollout_steps += 1

        dst = np.linalg.norm(self.goal_space_projection(self.state) - self.goal_space_projection(self.goal))
        info = {'target_reached': dst<= self.width_test}
        #reward = -dst

        reward = 0.

        return OrderedDict([
                ("observation", self.state.copy()),
                ("achieved_goal", self.goal_space_projection(self.state).copy()),
                ("desired_goal", self.goal_space_projection(self.goal).copy()),]), reward, done, info

    def state_vector(self):
        return self.state

    def _get_obs(self):
        return OrderedDict(
            [
                ("observation", self.state.copy()),
                ("achieved_goal", self.goal_space_projection(self.state).copy()),
                ("desired_goal", self.goal_space_projection(self.goal).copy()),
            ]
        )

    def goal_vector(self):
        return self.goal

    def set_state(self, starting_state):
        self.state = np.array(starting_state)
        return self.state_vector()

    def set_goal_state(self, goal_state):
        self.goal = self.goal_space_projection(np.array(goal_state))
        return 0

    def goal_space_projection(self, obs):
        return obs[:2]

    def sample_task(self):
        """
        Sample task for low-level policy training
        """

        delta_step = 1

        ## make sure the ratio can be computed (at least a success or a failure)
        weights_available = True
        for i in range(0,len(self.L_runs_results)- delta_step ):
            if len(self.L_runs_results[i]) == 0:
                weights_available = False

        # print("weights_available = ", weights_available)

        ## weighted sampling of the starting state
        if self.weighted_selection and weights_available:
            L_weights = []
            for i in range(0, len(self.L_states) - delta_step ):
                # print("self.L_runs_success[i] = ", self.L_runs_success[i])
                # print("self.L_runs_fails[i] = ", self.L_runs_fails[i])

                # print("self.L_runs_results[i] = ", self.L_runs_results[i])

                nb_runs_success = self.L_runs_results[i].count(1)
                nb_runs_fails = self.L_runs_results[i].count(0)

                s_r = float(nb_runs_success/(nb_runs_fails + nb_runs_success))

                ## on cape l'inversion
                if s_r <= 0.1:
                    s_r = 10
                else:
                    s_r = 1./s_r

                L_weights.append(s_r)

            max = sum(L_weights)
            pick = random.uniform(0, max)

            current = 0
            for i in range(0,len(L_weights)):
                s_r = L_weights[i]
                current += s_r
                if current > pick:
                    break

            self.indx_start = i

        # targeted selection
        elif self.target_selection :

            pick = np.random.random()

            if self.model != None :
                if pick < self.target_ratio and self.model.last_failed_task != None:
                    print("\ntarget sampling")
                    self.indx_start = self.model.last_failed_task
                else:
                    self.indx_start = random.randint(0, len(self.L_states) - delta_step - 1 )
            else:
                self.indx_start = random.randint(0, len(self.L_states) - delta_step - 1 )

        ## uniform selection
        else:
            self.indx_start = random.randint(0, len(self.L_states) - delta_step - 1 )


        pick = np.random.random()

        if self.allow_skipping and pick < self.skipping_ratio and self.indx_start  < len(self.L_states) - delta_step - 2:
            delta_step += 1
            self.skipping = True

        self.indx_goal = self.indx_start + delta_step
        length_task = sum(self.L_steps[self.indx_start:self.indx_goal]) #+ 5  ## bonus timesteps pour compenser les starting states reculés.
        # starting_state = self.L_states[self.indx_start]
        starting_state = random.choice(self.starting_state_set[self.indx_start])

        goal_state = self.L_states[self.indx_goal]

        return starting_state, length_task, goal_state


    def reset(self):
        # print("state = ", self.state_vector())
        self.reset_primitive()
        self.testing = False
        self.skipping = False

        starting_state, length_task, goal_state = self.sample_task()

        self.set_goal_state(goal_state)
        self.set_state(starting_state)
        self.max_steps = length_task
        #self.max_steps = 10

        self.rollout_steps = 0
        self.traj = []
        self.first_part_trans = []

        self.traj.append(self.state_vector())

        return OrderedDict([
                ("observation", self.state.copy()),
                ("achieved_goal", self.goal_space_projection(self.state.copy())),
                ("desired_goal", self.goal_space_projection(self.goal.copy())),])


# class DubinsMazeEnvGCPHERSB3_clean(DubinsMazeEnv):
#
#     def __init__(self, L_states, L_steps, args={
#             'mazesize':5,
#             'random_seed':0,
#             'mazestandard':False,
#             'wallthickness':0.1,
#             'wallskill':True,
#             'targetkills':True,
#             'max_steps': 50,
#             'width': 0.1
#         }):
#
#         super(DubinsMazeEnvGCPHERSB3_clean,self).__init__(args = args)
#
#         print("state = ", self.state)
#
#         self.max_steps = args['max_steps']
#         # Counter of steps per episode
#         self.args = args
#         self.rollout_steps = 0
#
#         self.action_space = spaces.Box(np.array([0., -1.]), np.array([1., 1.]), dtype = np.float32)
#
#         ms = int(args['mazesize'])
#         self.observation_space = spaces.Dict(
#                 {
#                     "observation": spaces.Box(
#                         low = np.array([0,0,-4]),
#                         high = np.array([ms,ms,4])),
#                     "achieved_goal": spaces.Box(
#                         low = np.array([0,0]),
#                         high = np.array([ms,ms])),
#                     "desired_goal": spaces.Box(
#                         low = np.array([0,0]),
#                         high = np.array([ms,ms])),
#                 }
#             )
#
#         self.state =  np.array([0.5, 0.5, 0.])
#         self.goal= np.array([args['mazesize']-0.5, args['mazesize']-0.5])
#         # self.width = args['width']
#         self.width_reward = 0.3
#         self.width_success = 0.05
#         self.width = self.width_success
#         self.width_test = self.width_success
#         self.L_states = L_states
#         self.L_steps = L_steps
#         self.total_steps = 0
#         self.starting_state_set = []
#         self.L_runs_results = []
#         self.results_size = 10
#
#         self.success_bonus = 10
#         self.traj = []
#         self.indx_start = 0
#         self.indx_goal = -1
#         self.testing = False
#         self.expanded = False
#
#         self.buffer_transitions = []
#
#         self.bonus = True
#         self.weighted_selection = False
#
#         self.target_selection = False
#         self.target_ratio = 0.3
#
#         self.skipping_ratio = 0.25
#         self.skipping = False
#         self.allow_skipping = False
#         self.skipping_results = []
#
#         self.frame_skip = 3
#
#         self.skipping_success_ratio = 0.75
#
#     def state_act(self, action):
#         r = np.copy(self.state)
#         return self.update_state(r, action, self.delta_t)
#
#
#     def compute_reward(self, achieved_goal, desired_goal, info):
#
#         if len(achieved_goal.shape) ==  1:
#             dst = np.linalg.norm(self.goal_space_projection(achieved_goal) - self.goal_space_projection(desired_goal), axis = -1)
#             #_info = {'target_reached': dst<= self.width_success}
#             _info = {'reward_boolean': dst<= self.width_reward}
#
#             if _info['reward_boolean']:
#                 #return -0.1
#                 return 0.
#
#             else:
#                 return -1.
#
#         else:
#             ## compute -1 or 0 reward
#             distances = np.linalg.norm(achieved_goal[:,:2] - desired_goal[:,:2], axis=-1)  ## pour éviter le cas ou dist == 0.
#             distances_mask = (distances <= self.width_reward).astype(np.float32)
#
#             # rewards = distances_mask - 1. # {-1, 0.}
#             # rewards = distances_mask - 1. - distances_mask * 0.2 # {-1, -0.1}
#             rewards = distances_mask - 1. #- distances_mask * 0.1 # {-1, -0.1}
#
#             return rewards
#
#
    def step(self, action) :

        state = self.state_vector()

        for i in range(self.frame_skip):
            new_state, reward, done, info = self._step(action)
            self.traj.append(new_state)

        self.rollout_steps += 1

        dst = np.linalg.norm(self.goal_space_projection(self.state) - self.goal_space_projection(self.goal))
        info = {'target_reached': dst<= self.width_success}

        info['goal_indx'] = copy.deepcopy(self.indx_goal)
        info['goal'] = copy.deepcopy(self.goal)

        if info['target_reached']: # achieved goal

            self.L_runs_results[self.indx_start].append(1)

            if len(self.L_runs_results[self.indx_start]) > self.results_size:
                self.L_runs_results[self.indx_start].pop(0)

            assert len(self.L_runs_results[self.indx_start]) <= self.results_size

            done = True

            reward = self.compute_reward(self.state, self.goal, info)

            # if self.indx_goal < len(self.starting_state_set) :
            #     ## add new sample to the set of starting_states
            #     self.starting_state_set[self.indx_goal].append(new_state)

            ## UPDATE SKIPPING RESULTS
            if self.allow_skipping and self.skipping == True:
                self.skipping_results[self.indx_goal].append(1)

                if len(self.skipping_results[self.indx_goal]) > self.results_size:

                    self.skipping_results[self.indx_goal].pop(0)

                    nb_skip_success = self.skipping_results[self.indx_goal].count(1)
                    nb_skip_fails = self.skipping_results[self.indx_goal].count(0)

                    s_r = float(nb_skip_success/(nb_skip_fails + nb_skip_success))
                    assert s_r >= 0. and s_r <= 1.
                    print("success_ratio skip nb " + str(self.indx_goal) + " = " + str(s_r))

                    if s_r > self.skipping_success_ratio :
                        # remove starting state
                        self.starting_state_set.pop(self.indx_goal-1)
                        self.L_states.pop(self.indx_goal-1)

                        print("self.indx_start = ", self.indx_start)
                        print("self.indx_goal -1  = ", self.indx_goal -1)
                        print("self.L_steps[self.indx_start] = ", self.L_steps[self.indx_start])
                        print("self.L_steps[self.indx_goal-1] = ", self.L_steps[self.indx_goal-1])
                        print("self.L_steps = ", self.L_steps)

                        # remove steps after adding them to the kept starting state
                        self.L_steps[self.indx_start] += self.L_steps[self.indx_goal-1]
                        print("self.L_steps[self.indx_start] = ", self.L_steps[self.indx_start])

                        self.L_steps.pop(self.indx_goal-1)
                        print("self.L_steps = ", self.L_steps)

                        assert sum(self.L_steps) == self.total_steps # make sure we kept enough timesteps

                        # empty results buffer for the successfuly reached goal and the next goal
                        if self.indx_goal + 1 < len(self.skipping_results):
                            self.skipping_results[self.indx_goal+1] = []
                        self.skipping_results[self.indx_goal] = []

                        # remove skipping results for the skipped index
                        self.skipping_results.pop(self.indx_goal-1)

                        # make sure we removed everything properly
                        assert len(self.L_states) == len(self.L_steps) and len(self.L_steps) == len(self.skipping_results)


            #prev_goal = copy.deepcopy(self.goal)
            prev_goal = self.goal_space_projection(self.goal).copy()
            info['done'] = done
            info['goal'] = self.goal_space_projection(self.goal).copy()
            info['traj'] = self.traj
            self.reset()
            return OrderedDict([
                    ("observation", self.state.copy()),
                    ("achieved_goal", self.goal_space_projection(self.state).copy()),
                    ("desired_goal", prev_goal)]), reward, done, info

        elif self.rollout_steps >= self.max_steps:

            self.L_runs_results[self.indx_start].append(0)

            if len(self.L_runs_results[self.indx_start]) > self.results_size:
                self.L_runs_results[self.indx_start].pop(0)

            ## UPDATE SKIPPING RESULTS (NO SKIPPING CONFIRMATION HERE)
            if self.allow_skipping and self.skipping == True:
                print("self.indx_goal = ", self.indx_goal)
                print("len(self.skipping_results) = ", len(self.skipping_results))
                self.skipping_results[self.indx_goal].append(0)

                if len(self.skipping_results[self.indx_goal]) > self.results_size:
                    self.skipping_results[self.indx_goal].pop(0)

            reward = self.compute_reward(self.state, self.goal, info)


            if self.indx_goal < len(self.starting_state_set) :
                ## add new starting_state
                to_add = True

                achieved_goal = new_state[:2].copy()

                for state in self.starting_state_set[self.indx_goal]:

                    goal = np.array(state[:2]).copy()
                    #print("achieved_goal = ", achieved_goal)
                    #print("goal = ", goal)

                    if np.linalg.norm(achieved_goal - goal, axis = -1) < self.width_success:
                        to_add = False

                        break

                if to_add :
                    self.starting_state_set[self.indx_goal].append(new_state)


            done = True
            # prev_goal = copy.deepcopy(self.goal)
            prev_goal = self.goal_space_projection(self.goal).copy()
            info['done'] = done
            info['goal'] = self.goal_space_projection(self.goal).copy()
            info['traj'] = self.traj
            self.reset()

            return OrderedDict([
                    ("observation", self.state.copy()),
                    ("achieved_goal", self.goal_space_projection(self.state).copy()),
                    ("desired_goal", prev_goal)]), reward, done, info

        else:

            done = False

            reward = self.compute_reward(self.state, self.goal, info)

            info['done'] = done
            return OrderedDict([
                    ("observation", self.state.copy()),
                    ("achieved_goal", self.goal_space_projection(self.state).copy()),
                    ("desired_goal", self.goal_space_projection(self.goal).copy()),]), reward, done, info

#     def step_test(self, action) :
#
#         obs = self.state_vector()
#
#         for i in range(self.frame_skip):
#             new_obs, reward, done, info = self._step(action)
#             self.traj.append(new_obs[:2])
#
#         self.rollout_steps += 1
#
#         dst = np.linalg.norm(self.goal_space_projection(self.state) - self.goal_space_projection(self.goal))
#         info = {'target_reached': dst<= self.width_test}
#         #reward = -dst
#
#         reward = 0.
#
#         return OrderedDict([
#                 ("observation", self.state.copy()),
#                 ("achieved_goal", self.goal_space_projection(self.state).copy()),
#                 ("desired_goal", self.goal_space_projection(self.goal).copy()),]), reward, done, info
#
#     def state_vector(self):
#         return self.state
#
#     def _get_obs(self):
#         return OrderedDict(
#             [
#                 ("observation", self.state.copy()),
#                 ("achieved_goal", self.goal_space_projection(self.state).copy()),
#                 ("desired_goal", self.goal_space_projection(self.goal).copy()),
#             ]
#         )
#
#     def goal_vector(self):
#         return self.goal
#
#     def set_state(self, starting_state):
#         self.state = np.array(starting_state)
#         return self.state_vector()
#
#     def set_goal_state(self, goal_state):
#         self.goal = self.goal_space_projection(np.array(goal_state))
#         return 0
#
#     def goal_space_projection(self, obs):
#         return obs[:2]
#
#     def sample_task(self, eval):
#         """
#         Sample task for low-level policy training
#         """
#
#         delta_step = 1
#
#         ## make sure the ratio can be computed (at least a success or a failure)
#         weights_available = True
#         for i in range(0,len(self.L_runs_results)- delta_step ):
#             if len(self.L_runs_results[i]) == 0:
#                 weights_available = False
#
#         # print("weights_available = ", weights_available)
#
#         ## weighted sampling of the starting state
#         if self.weighted_selection and weights_available:
#             L_weights = []
#             for i in range(0, len(self.L_states) - delta_step ):
#                 # print("self.L_runs_success[i] = ", self.L_runs_success[i])
#                 # print("self.L_runs_fails[i] = ", self.L_runs_fails[i])
#
#                 # print("self.L_runs_results[i] = ", self.L_runs_results[i])
#
#                 nb_runs_success = self.L_runs_results[i].count(1)
#                 nb_runs_fails = self.L_runs_results[i].count(0)
#
#                 s_r = float(nb_runs_success/(nb_runs_fails + nb_runs_success))
#
#                 ## on cape l'inversion
#                 if s_r <= 0.1:
#                     s_r = 10
#                 else:
#                     s_r = 1./s_r
#
#                 L_weights.append(s_r)
#
#             max = sum(L_weights)
#             pick = random.uniform(0, max)
#
#             current = 0
#             for i in range(0,len(L_weights)):
#                 s_r = L_weights[i]
#                 current += s_r
#                 if current > pick:
#                     break
#
#             self.indx_start = i
#
#         # targeted selection
#         elif self.target_selection :
#
#             pick = np.random.random()
#
#             if self.model != None :
#                 if pick < self.target_ratio and self.model.last_failed_task != None:
#                     print("\ntarget sampling")
#                     self.indx_start = self.model.last_failed_task
#                 else:
#                     self.indx_start = random.randint(0, len(self.L_states) - delta_step - 1 )
#             else:
#                 self.indx_start = random.randint(0, len(self.L_states) - delta_step - 1 )
#
#         ## uniform selection
#         else:
#             self.indx_start = random.randint(0, len(self.L_states) - delta_step - 1 )
#
#
#         pick = np.random.random()
#
#         if self.allow_skipping and pick < self.skipping_ratio and self.indx_start  < len(self.L_states) - delta_step - 2:
#             delta_step += 1
#             self.skipping = True
#
#         self.indx_goal = self.indx_start + delta_step
#         length_task = sum(self.L_steps[self.indx_start:self.indx_goal]) #+ 5  ## bonus timesteps pour compenser les starting states reculés.
#         # starting_state = self.L_states[self.indx_start]
#         #starting_state = random.choice(self.starting_state_set[self.indx_start])
#         #starting_state = self.starting_state_set[self.indx_start][0]
#
#         ## diversify starting state and goal state
#
#         ## crashtest
#         starting_state = self.starting_state_set[self.indx_start][0]
#
#         if self.indx_goal < len(self.L_states) - 1:
#             goal_state = random.choice(self.starting_state_set[self.indx_goal])
#         else:
#             goal_state = self.starting_state_set[self.indx_goal][0]
#
#
#         ## crashtest
#         # random_diversity_pick = np.random.random()
#         #
#         # if random_diversity_pick > 0.5: ## start from diverse starting state and aim at demonstration goal
#         #
#         #     if self.indx_start > 0 and self.indx_start < len(self.L_states) - 2:
#         #         starting_state = random.choice(self.starting_state_set[self.indx_start])
#         #     else:
#         #         starting_state = self.starting_state_set[self.indx_start][0]
#         #
#         #     goal_state = self.starting_state_set[self.indx_goal][0]
#         #
#         # else: ## start from demonstration state and aim at diverse goals
#         #
#         #     starting_state = self.starting_state_set[self.indx_start][0]
#         #
#         #     if self.indx_goal < len(self.L_states) - 1:
#         #         goal_state = random.choice(self.starting_state_set[self.indx_goal])
#         #     else:
#         #         goal_state = self.starting_state_set[self.indx_goal][0]
#
#
#         ## toy example
#         # if self.indx_start == 1:
#         #     starting_state = random.choice(self.starting_state_set[self.indx_start])
#         # else:
#         #     starting_state = self.starting_state_set[self.indx_start][0]
#         #
#         # if self.indx_goal == 1:
#         #     goal_state = random.choice(self.starting_state_set[self.indx_goal])
#         # else:
#         #     goal_state = self.starting_state_set[self.indx_goal][0]
#
#         return starting_state, length_task, goal_state
#
#
#     def reset(self, eval = False):
#         # print("state = ", self.state_vector())
#         self.reset_primitive()
#         self.testing = False
#         self.skipping = False
#
#         starting_state, length_task, goal_state = self.sample_task(eval)
#
#         self.set_goal_state(goal_state)
#
#         #print("goal in env = ", self.goal)
#
#         self.set_state(starting_state)
#         self.max_steps = length_task
#         #self.max_steps = 10
#
#         self.rollout_steps = 0
#         self.traj = []
#         self.first_part_trans = []
#
#         self.traj.append(self.state_vector())
#
#         return OrderedDict([
#                 ("observation", self.state.copy()),
#                 ("achieved_goal", self.goal_space_projection(self.state.copy())),
#                 ("desired_goal", self.goal_space_projection(self.goal.copy())),])


# class DubinsMazeEnvGCPHERSB3_clean(DubinsMazeEnv):
#
#     def __init__(self, L_states, L_steps, args={
#             'mazesize':5,
#             'random_seed':0,
#             'mazestandard':False,
#             'wallthickness':0.1,
#             'wallskill':True,
#             'targetkills':True,
#             'max_steps': 50,
#             'width': 0.1
#         }):
#
#         super(DubinsMazeEnvGCPHERSB3_clean,self).__init__(args = args)
#
#         print("state = ", self.state)
#
#         self.max_steps = args['max_steps']
#         # Counter of steps per episode
#         self.args = args
#         self.rollout_steps = 0
#
#         self.action_space = spaces.Box(np.array([0., -1.]), np.array([1., 1.]), dtype = np.float32)
#
#         ms = int(args['mazesize'])
#         self.observation_space = spaces.Dict(
#                 {
#                     "observation": spaces.Box(
#                         low = np.array([0,0,-4]),
#                         high = np.array([ms,ms,4])),
#                     "achieved_goal": spaces.Box(
#                         low = np.array([0,0]),
#                         high = np.array([ms,ms])),
#                     "desired_goal": spaces.Box(
#                         low = np.array([0,0]),
#                         high = np.array([ms,ms])),
#                 }
#             )
#
#         self.state =  np.array([0.5, 0.5, 0.])
#         self.goal= np.array([args['mazesize']-0.5, args['mazesize']-0.5])
#         # self.width = args['width']
#         self.width_reward = 0.3
#         self.width_success = 0.3
#         self.width = self.width_success
#         self.width_test = self.width_success
#         self.L_states = L_states
#         self.L_steps = L_steps
#         self.total_steps = 0
#         self.starting_state_set = []
#         self.L_tasks_results = []
#         self.results_size = 10
#
#         self.success_bonus = 10
#         self.traj = []
#         self.indx_start = 0
#         self.indx_goal = -1
#         self.testing = False
#         self.expanded = False
#
#         self.buffer_transitions = []
#
#         self.bonus = True
#         self.weighted_selection = True
#
#         self.target_selection = False
#         self.target_ratio = 0.3
#
#         self.skipping_ratio = 0.25
#         self.skipping = False
#         self.allow_skipping = False
#         self.skipping_results = []
#
#         self.frame_skip = 1
#         # self.frame_skip = 3
#
#         self.skipping_success_ratio = 0.75
#
#     def state_act(self, action):
#         r = np.copy(self.state)
#         return self.update_state(r, action, self.delta_t)
#
#
#     def compute_reward(self, achieved_goal, desired_goal, info):
#
#         if len(achieved_goal.shape) ==  1:
#             dst = np.linalg.norm(self.goal_space_projection(achieved_goal) - self.goal_space_projection(desired_goal), axis = -1)
#             #_info = {'target_reached': dst<= self.width_success}
#             _info = {'reward_boolean': dst<= self.width_reward}
#
#             if _info['reward_boolean']:
#                 return -0.1
#                 # return 0.
#
#             else:
#                 return -1.
#
#         else:
#             ## compute -1 or 0 reward
#             distances = np.linalg.norm(achieved_goal[:,:2] - desired_goal[:,:2], axis=-1)  ## pour éviter le cas ou dist == 0.
#             distances_mask = (distances <= self.width_reward).astype(np.float32)
#
#             rewards = distances_mask - 1. - distances_mask * 0.1 # {-1, -0.1}
#             # rewards = distances_mask - 1.
#
#             return rewards
#
#
#     def step(self, action) :
#
#         state = self.state_vector()
#
#         for i in range(self.frame_skip):
#             new_state, reward, done, info = self._step(action)
#             #if not self.overshooting:
#             self.traj.append(new_state)
#
#         self.rollout_steps += 1
#
#         dst = np.linalg.norm(self.goal_space_projection(self.state) - self.goal_space_projection(self.goal))
#         info = {'target_reached': dst<= self.width_success}
#
#         info['goal_indx'] = copy.deepcopy(self.indx_goal)
#         info['goal'] = copy.deepcopy(self.goal)
#
#         if self.target_subgoal: ## if we aim at the subgoal, subgoal cannot be reached
#             #print("targeting subgoal")
#             info["subgoal_reached"] = False
#             info["subgoal"] = copy.deepcopy(self.goal) ## subgoal is goal if we still target the subgoal
#             info["subgoal_indx"] = self.indx_goal
#
#         if self.overshooting: ## if we reached the subgoal, start overshooting and keep track of subgoal used
#             #print("overshooting")
#             info["subgoal_reached"] = True
#             info["subgoal"] = copy.deepcopy(self.subgoal)
#             info["subgoal_indx"] = self.indx_goal - 1
#
#         if info['target_reached']: # achieved goal
#
#             if self.target_subgoal and self.indx_goal < len(self.L_states) - 1:
#
#                 self.L_tasks_results[self.indx_goal].append(1)
#
#                 if len(self.L_tasks_results[self.indx_goal]) > self.results_size:
#                     self.L_tasks_results[self.indx_goal].pop(0)
#
#                 #print("successfully reached subgoal")
#                 done = False
#                 self.target_subgoal = False
#                 self.overshooting = True
#                 self.subgoal = copy.deepcopy(self.goal)
#
#                 #info['subgoal_reached'] = True
#                 info["subgoal"] = copy.deepcopy(self.subgoal)
#
#                 ## overshoot for next goal
#                 self.indx_goal += 1
#                 goal_state = self.starting_state_set[self.indx_goal][0]
#                 self.set_goal_state(goal_state)
#                 self.rollout_steps = 0
#
#                 prev_goal = self.goal_space_projection(self.goal).copy()
#                 info['done'] = done
#                 info['goal'] = self.goal_space_projection(self.goal).copy()
#                 info['traj'] = self.traj
#
#             else:
#                 #print("successfully overshoot")
#                 done = True
#
#                 reward = self.compute_reward(self.state, self.goal, info)
#
#                 #prev_goal = copy.deepcopy(self.goal)
#                 prev_goal = self.goal_space_projection(self.goal).copy()
#                 info['done'] = done
#                 info['goal'] = self.goal_space_projection(self.goal).copy()
#                 info['traj'] = self.traj
#                 self.reset()
#
#             return OrderedDict([
#                     ("observation", self.state.copy()),
#                     ("achieved_goal", self.goal_space_projection(self.state).copy()),
#                     ("desired_goal", prev_goal)]), reward, done, info
#
#         elif self.rollout_steps >= self.max_steps:
#
#             self.L_tasks_results[self.indx_goal].append(0)
#
#             if len(self.L_tasks_results[self.indx_goal]) > self.results_size:
#                 self.L_tasks_results[self.indx_goal].pop(0)
#
#             #print("failed overshooting or subgoal reaching")
#             reward = self.compute_reward(self.state, self.goal, info)
#
#             done = True
#             # prev_goal = copy.deepcopy(self.goal)
#             prev_goal = self.goal_space_projection(self.goal).copy()
#             info['done'] = done
#             info['goal'] = self.goal_space_projection(self.goal).copy()
#             info['traj'] = self.traj
#             self.reset()
#
#             return OrderedDict([
#                     ("observation", self.state.copy()),
#                     ("achieved_goal", self.goal_space_projection(self.state).copy()),
#                     ("desired_goal", prev_goal)]), reward, done, info
#
#         else:
#
#             done = False
#
#             reward = self.compute_reward(self.state, self.goal, info)
#
#             info['done'] = done
#             return OrderedDict([
#                     ("observation", self.state.copy()),
#                     ("achieved_goal", self.goal_space_projection(self.state).copy()),
#                     ("desired_goal", self.goal_space_projection(self.goal).copy()),]), reward, done, info
#
#     def step_test(self, action) :
#
#         obs = self.state_vector()
#
#         for i in range(self.frame_skip):
#             new_obs, reward, done, info = self._step(action)
#             self.traj.append(new_obs[:2])
#
#         self.rollout_steps += 1
#
#         dst = np.linalg.norm(self.goal_space_projection(self.state) - self.goal_space_projection(self.goal))
#         info = {'target_reached': dst<= self.width_test}
#         #reward = -dst
#
#         reward = 0.
#
#         return OrderedDict([
#                 ("observation", self.state.copy()),
#                 ("achieved_goal", self.goal_space_projection(self.state).copy()),
#                 ("desired_goal", self.goal_space_projection(self.goal).copy()),]), reward, done, info
#
#     def state_vector(self):
#         return self.state
#
#     def _get_obs(self):
#         return OrderedDict(
#             [
#                 ("observation", self.state.copy()),
#                 ("achieved_goal", self.goal_space_projection(self.state).copy()),
#                 ("desired_goal", self.goal_space_projection(self.goal).copy()),
#             ]
#         )
#
#     def goal_vector(self):
#         return self.goal
#
#     def set_state(self, starting_state):
#         self.state = np.array(starting_state)
#         return self.state_vector()
#
#     def set_goal_state(self, goal_state):
#         self.goal = self.goal_space_projection(np.array(goal_state))
#         return 0
#
#     def goal_space_projection(self, obs):
#         return obs[:2]
#
#     def sample_task(self, eval):
#         """
#         Sample task for low-level policy training
#         """
#
#         delta_step = 1
#
#
#         weights_available = True
#         for i in range(0,len(self.L_tasks_results)- delta_step ):
#             if len(self.L_tasks_results[i]) == 0:
#                 weights_available = False
#
#         if self.weighted_selection and weights_available:
#             L_weights = []
#             for i in range(0, len(self.L_states) - delta_step ):
#
#                 nb_tasks_success = self.L_tasks_results[i].count(1)
#
#                 s_r = float(nb_tasks_success/len(self.L_tasks_results[i]))
#
#                 ## on cape l'inversion
#                 if s_r <= 0.1:
#                     s_r = 10
#                 else:
#                     s_r = 1./s_r
#
#                 L_weights.append(s_r)
#
#             max = sum(L_weights)
#             pick = random.uniform(0, max)
#
#             current = 0
#             for i in range(0,len(L_weights)):
#                 s_r = L_weights[i]
#                 current += s_r
#                 if current > pick:
#                     break
#
#             self.indx_start = i
#             self.indx_goal = self.indx_start + delta_step
#             length_task = sum(self.L_steps[self.indx_start:self.indx_goal])
#
#             starting_state = self.starting_state_set[self.indx_start][0]
#
#             ## diverse choice of goal
#             if self.indx_goal < len(self.L_states) - 1:
#                 goal_state = random.choice(self.starting_state_set[self.indx_goal])
#             else:
#                 goal_state = self.starting_state_set[self.indx_goal][0]
#
#             # goal_state = self.starting_state_set[self.indx_goal][0]
#
#
#         else:
#             self.indx_start = random.randint(0, len(self.L_states) - delta_step - 1 )
#             self.indx_goal = self.indx_start + delta_step
#             length_task = sum(self.L_steps[self.indx_start:self.indx_goal]) #+ 5  ## bonus timesteps pour compenser les starting states reculés.
#
#             ## diversify starting state and goal state
#
#             ## crashtest
#             starting_state = self.starting_state_set[self.indx_start][0]
#
#             # if self.indx_goal < len(self.L_states) - 1:
#             #     goal_state = random.choice(self.starting_state_set[self.indx_goal])
#             # else:
#             #     goal_state = self.starting_state_set[self.indx_goal][0]
#
#             goal_state = self.starting_state_set[self.indx_goal][0]
#
#         ## crashtest
#         # random_diversity_pick = np.random.random()
#         #
#         # if random_diversity_pick > 0.5: ## start from diverse starting state and aim at demonstration goal
#         #
#         #     if self.indx_start > 0 and self.indx_start < len(self.L_states) - 2:
#         #         starting_state = random.choice(self.starting_state_set[self.indx_start])
#         #     else:
#         #         starting_state = self.starting_state_set[self.indx_start][0]
#         #
#         #     goal_state = self.starting_state_set[self.indx_goal][0]
#         #
#         # else: ## start from demonstration state and aim at diverse goals
#         #
#         #     starting_state = self.starting_state_set[self.indx_start][0]
#         #
#         #     if self.indx_goal < len(self.L_states) - 1:
#         #         goal_state = random.choice(self.starting_state_set[self.indx_goal])
#         #     else:
#         #         goal_state = self.starting_state_set[self.indx_goal][0]
#
#
#         ## toy example
#         # if self.indx_start == 1:
#         #     starting_state = random.choice(self.starting_state_set[self.indx_start])
#         # else:
#         #     starting_state = self.starting_state_set[self.indx_start][0]
#         #
#         # if self.indx_goal == 1:
#         #     goal_state = random.choice(self.starting_state_set[self.indx_goal])
#         # else:
#         #     goal_state = self.starting_state_set[self.indx_goal][0]
#
#         return starting_state, length_task, goal_state
#
#
#     def reset(self, eval = False):
#         # print("state = ", self.state_vector())
#         self.reset_primitive()
#         self.testing = False
#         self.skipping = False
#
#         #print("\n reset")
#
#         starting_state, length_task, goal_state = self.sample_task(eval)
#
#         self.set_goal_state(goal_state)
#         self.set_state(starting_state)
#
#         self.target_subgoal = True
#         self.overshooting = False
#
#         self.max_steps = length_task
#         #self.max_steps = 10
#
#         self.rollout_steps = 0
#         self.traj = []
#         self.first_part_trans = []
#
#         self.traj.append(self.state_vector())
#
#         return OrderedDict([
#                 ("observation", self.state.copy()),
#                 ("achieved_goal", self.goal_space_projection(self.state.copy())),
#                 ("desired_goal", self.goal_space_projection(self.goal.copy())),])

class DubinsMazeEnvGCPHERSB3_clean(DubinsMazeEnv):

    def __init__(self, L_full_demonstration, L_states, L_goals, L_inner_states, L_budgets, env_option = "5"):

        args={
                'mazesize':int(env_option),
                'random_seed':0,
                'mazestandard':False,
                'wallthickness':0.1,
                'wallskill':True,
                'targetkills':True,
                'max_steps': 50,
                'width': 0.1
            }

        super(DubinsMazeEnvGCPHERSB3_clean,self).__init__(args = args)

        print("state = ", self.state)

        ## tasks
        self.tasks = TasksManager(L_full_demonstration, L_states, L_goals, L_inner_states, L_budgets, env_option)

        self.max_steps = args['max_steps']
        # Counter of steps per episode
        self.args = args
        self.rollout_steps = 0

        self.action_space = spaces.Box(np.array([0., -1.]), np.array([1., 1.]), dtype = np.float32)

        ms = int(args['mazesize'])
        self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(
                        low = np.array([0,0,-4]),
                        high = np.array([ms,ms,4])),
                    "achieved_goal": spaces.Box(
                        low = np.array([0,0]),
                        high = np.array([ms,ms])),
                    "desired_goal": spaces.Box(
                        low = np.array([0,0]),
                        high = np.array([ms,ms])),
                }
            )

        self.state =  np.array([0.5, 0.5, 0.])
        self.goal= np.array([args['mazesize']-0.5, args['mazesize']-0.5])
        # self.width = args['width']
        self.width_reward = 0.3
        self.width_success = 0.3
        self.width = self.width_success
        self.width_test = self.width_success
        self.total_steps = 0
        self.results_size = 10

        self.traj = []

        self.testing = False
        self.expanded = False

        self.buffer_transitions = []

        self.target_ratio = 0.3

        self.frame_skip = 2


    def state_act(self, action):
        r = np.copy(self.state)
        return self.update_state(r, action, self.delta_t)

    def compute_distance_in_goal_space(self, goal1, goal2):

        goal1 = np.array(goal1)
        goal2 = np.array(goal2)

        if len(goal1.shape) ==  1:
            return np.linalg.norm(goal1 - goal2, axis=-1)
        else:
            return np.linalg.norm(goal1[:,:] - goal2[:,:], axis=-1)

    def compute_reward(self, achieved_goal, desired_goal, info):

        if len(achieved_goal.shape) ==  1:
            # dst = np.linalg.norm(achieved_goal - desired_goal, axis = -1)
            dst = self.compute_distance_in_goal_space(achieved_goal, desired_goal)

            # print("dst = ", dst)

            #_info = {'target_reached': dst<= self.width_success}
            _info = {'reward_boolean': dst<= self.width_reward}

            if _info['reward_boolean']:
                return -0.1
                # return 0.
            else:
                return -1.

        else:
            ## compute -1 or 0 reward
            # distances = np.linalg.norm(achieved_goal[:,:] - desired_goal[:,:], axis=-1)  ## pour éviter le cas ou dist == 0.

            distances = self.compute_distance_in_goal_space(achieved_goal, desired_goal)
            distances_mask = (distances <= self.width_reward).astype(np.float32)

            rewards = distances_mask - 1. - distances_mask * 0.1 # {-1, -0.1}
            # rewards = distances_mask - 1.

            return rewards

    def step(self, action) :

        state = self.get_state()

        for i in range(self.frame_skip):
            new_state, reward, done, info = self._step(action)
            new_inner_state = new_state.copy()

            #if not self.overshooting:
            self.traj.append(new_state)

            if self.tasks.subgoal_adaptation:
                self.tasks.add_new_starting_state(self.tasks.indx_goal, new_inner_state, new_state)

        self.rollout_steps += 1

        dst = np.linalg.norm(self.project_to_goal_space(new_state) - self.goal)
        info = {'target_reached': dst<= self.width_success}

        info['goal_indx'] = copy.deepcopy(self.tasks.indx_goal)
        info['goal'] = copy.deepcopy(self.goal)

        if self.target_subgoal: ## if we aim at the subgoal, subgoal cannot be reached
        #print("targeting subgoal")
            info["subgoal_reached"] = False
            info["subgoal"] = copy.deepcopy(self.goal_state) ## subgoal is goal if we still target the subgoal
            info["subgoal_indx"] = self.tasks.indx_goal

        if self.overshooting: ## if we reached the subgoal, start overshooting and keep track of subgoal used
            #print("overshooting")
            info["subgoal_reached"] = True
            info["subgoal"] = copy.deepcopy(self.subgoal_state)
            info["subgoal_indx"] = self.tasks.indx_goal - self.tasks.delta_step

        if info['target_reached']: # achieved goal
            if self.target_subgoal and self.tasks.indx_goal < len(self.tasks.L_states) - 1:
                self.tasks.add_success(self.tasks.indx_goal)

                #print("successfully reached subgoal")
                done = False
                reward = self.compute_reward(self.project_to_goal_space(new_state), self.goal, info)
                # print("reward = ", reward)

                self.target_subgoal = False
                self.overshooting = True
                self.subgoal = copy.deepcopy(self.goal)
                self.subgoal_state = copy.deepcopy(self.goal_state)

                #info['subgoal_reached'] = True
                info["subgoal"] = copy.deepcopy(self.subgoal)
                prev_goal = self.goal.copy()

                ## overshoot for next goal
                advance_bool = self.advance_task()

                if advance_bool:
                    self.rollout_steps = 0

                    info['done'] = done
                    info['goal'] = self.goal.copy()
                    info['traj'] = self.traj

                else:
                    done = True
                    info['done'] = done
                    info['goal'] = self.goal.copy()
                    info['traj'] = self.traj

                    self.reset()

            elif self.overshooting:
                #print("successfully overshoot")
                done = True

                reward = self.compute_reward(self.project_to_goal_space(new_state), self.goal, info)
                # print("reward = ", reward)

                #prev_goal = copy.deepcopy(self.goal)
                prev_goal = self.goal.copy()

                info['done'] = done
                info['goal'] = self.goal.copy()
                info['traj'] = self.traj

                ## update subgoal trial as success if successful overshoot
                if self.tasks.subgoal_adaptation:
                    self.tasks.update_overshoot_result(info["subgoal_indx"], info["subgoal"], True)

                self.reset()

            else:
                done = True

                reward = self.compute_reward(self.project_to_goal_space(new_state), self.goal, info)
                # print("reward = ", reward)

                #prev_goal = copy.deepcopy(self.goal)
                prev_goal = self.goal.copy()

                info['done'] = done
                info['goal'] = self.goal.copy()
                info['traj'] = self.traj

                self.reset()

            return OrderedDict([
                    ("observation", new_state.copy()), ## TODO: what's the actual state?
                    ("achieved_goal", self.project_to_goal_space(new_state).copy()),
                    ("desired_goal", prev_goal)]), reward, done, info


        elif self.rollout_steps >= self.max_steps:

            ## add failure to task results
            self.tasks.add_failure(self.tasks.indx_goal)

            #print("failed overshooting or subgoal reaching")
            reward = self.compute_reward(self.project_to_goal_space(new_state), self.goal, info)
            # print("reward = ", reward)

            done = True
            # prev_goal = copy.deepcopy(self.goal)
            prev_goal = self.goal.copy()
            info['done'] = done
            info['goal'] = self.goal.copy()
            info['traj'] = self.traj

            ## add failure to overshoot result
            if self.tasks.subgoal_adaptation:
                self.tasks.update_overshoot_result(info["subgoal_indx"], info["subgoal"], False)

            self.reset()

            return OrderedDict([
                    ("observation", new_state.copy()),
                    ("achieved_goal", self.project_to_goal_space(new_state).copy()),
                    ("desired_goal", prev_goal)]), reward, done, info

        else:

            done = False

            reward = self.compute_reward(self.project_to_goal_space(new_state), self.goal, info)

            info['done'] = done
            return OrderedDict([
                    ("observation", new_state.copy()),
                    ("achieved_goal", self.project_to_goal_space(new_state).copy()),
                    ("desired_goal", self.goal.copy()),]), reward, done, info

    def step_test(self, action) :

        for i in range(self.frame_skip):
            new_state, reward, done, info = self._step(action)
            self.traj.append(new_state)

        self.rollout_steps += 1

        dst = np.linalg.norm(self.project_to_goal_space(new_state) - self.goal)
        info = {'target_reached': dst<= self.width_test}
        #reward = -dst

        reward = 0.

        return OrderedDict([
        ("observation", new_state.copy()),
        ("achieved_goal", self.project_to_goal_space(new_state).copy()),
        ("desired_goal", self.goal.copy()),]), reward, done, info

    def get_state(self):
        return self.state

    def _get_obs(self):

        state = self.get_state()
        achieved_goal = self.project_to_goal_space(state)

        return OrderedDict(
        [
        ("observation", state.copy()),
        ("achieved_goal", achieved_goal.copy()),
        ("desired_goal", self.goal.copy()),
        ]
        )

    def goal_vector(self):
        return self.goal

    def set_state(self, state):
        self.state = np.array(state)
        return self.get_state()

    def set_goal_state(self, goal_state):
        self.goal_state = np.array(goal_state)
        self.goal = self.project_to_goal_space(goal_state)
        return 0

    def project_to_goal_space(self, state):
        return np.array(state[:2])

    def select_task(self):
        """
        Sample task for low-level policy training.
        """
        return self.tasks.select_task()

    def reset_task_by_nb(self, task_nb):

        self.reset()

        starting_state, length_task, goal_state = self.tasks.get_task(task_nb)

        self.set_goal_state(goal_state)
        self.set_state(starting_state)
        self.max_steps = length_task
        return

    def advance_task(self):
        print("self.goal = ", self.goal)
        goal_state, length_task, advance_bool = self.tasks.advance_task()

        if advance_bool:

            self.set_goal_state(goal_state)
            self.max_steps = length_task
        print("self.goal = ", self.goal)
        return advance_bool


    def reset(self, eval = False):

        self.testing = False
        self.skipping = False

        starting_state, length_task, goal_state = self.select_task()

        self.set_goal_state(goal_state)
        self.set_state(starting_state)

        self.target_subgoal = True
        self.overshooting = False

        self.max_steps = length_task
        #self.max_steps = 10

        self.rollout_steps = 0
        self.traj = []

        state = self.get_state()
        self.traj.append(state)

        return OrderedDict([
        ("observation", state.copy()),
        ("achieved_goal", self.project_to_goal_space(state).copy()),
        ("desired_goal", self.goal.copy()),])

class FreeDubinsMazeEnvGCPHERSB3_clean(FreeDubinsMazeEnv):

    def __init__(self, L_states, L_steps, args={
            'mazesize':5,
            'random_seed':0,
            'mazestandard':False,
            'wallthickness':0.1,
            'wallskill':True,
            'targetkills':True,
            'max_steps': 50,
            'width': 0.1
        }):

        super(FreeDubinsMazeEnvGCPHERSB3_clean,self).__init__(args = args)

        print("state = ", self.state)

        self.max_steps = args['max_steps']
        # Counter of steps per episode
        self.args = args
        self.rollout_steps = 0

        self.action_space = spaces.Box(np.array([0.1, -1.]), np.array([1., 1.]), dtype = np.float32)

        ms = int(args['mazesize'])
        self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(
                        low = np.array([0,0,-4]),
                        high = np.array([ms,ms,4])),
                    "achieved_goal": spaces.Box(
                        low = np.array([0,0]),
                        high = np.array([ms,ms])),
                    "desired_goal": spaces.Box(
                        low = np.array([0,0]),
                        high = np.array([ms,ms])),
                }
            )

        self.state =  np.array([0.5, 0.5, 0.])
        self.goal= np.array([args['mazesize']-0.5, args['mazesize']-0.5])
        # self.width = args['width']
        self.width_reward = 0.3
        self.width_success = 0.05
        self.width = self.width_success
        self.width_test = self.width_success
        self.L_states = L_states
        self.L_steps = L_steps
        self.total_steps = 0
        self.starting_state_set = []
        self.L_runs_results = []
        self.results_size = 10

        self.success_bonus = 10
        self.traj = []
        self.indx_start = 0
        self.indx_goal = -1
        self.testing = False
        self.expanded = False

        self.buffer_transitions = []

        self.bonus = True
        self.weighted_selection = False

        self.target_selection = False
        self.target_ratio = 0.3

        self.skipping_ratio = 0.25
        self.skipping = False
        self.allow_skipping = False
        self.skipping_results = []

        self.frame_skip = 3

        self.skipping_success_ratio = 0.75

    def state_act(self, action):
        r = np.copy(self.state)
        return self.update_state(r, action, self.delta_t)


    def compute_reward(self, achieved_goal, desired_goal, info):

        if len(achieved_goal.shape) ==  1:
            dst = np.linalg.norm(self.goal_space_projection(achieved_goal) - self.goal_space_projection(desired_goal), axis = -1)
            #_info = {'target_reached': dst<= self.width_success}
            _info = {'reward_boolean': dst<= self.width_reward}

            if _info['reward_boolean']:
                #return -0.1
                return 0.

            else:
                return -1.

        else:
            ## compute -1 or 0 reward
            distances = np.linalg.norm(achieved_goal[:,:2] - desired_goal[:,:2], axis=-1)  ## pour éviter le cas ou dist == 0.
            distances_mask = (distances <= self.width_reward).astype(np.float32)

            # rewards = distances_mask - 1. # {-1, 0.}
            # rewards = distances_mask - 1. - distances_mask * 0.2 # {-1, -0.1}
            rewards = distances_mask - 1. #- distances_mask * 0.1 # {-1, -0.1}

            return rewards


    def step(self, action) :

        state = self.state_vector()

        for i in range(self.frame_skip):
            new_state, reward, done, info = self._step(action)
            self.traj.append(new_state)

        self.rollout_steps += 1

        dst = np.linalg.norm(self.goal_space_projection(self.state) - self.goal_space_projection(self.goal))
        info = {'target_reached': dst<= self.width_success}

        info['goal_indx'] = self.indx_goal

        if info['target_reached']: # achieved goal

            self.L_runs_results[self.indx_start].append(1)

            if len(self.L_runs_results[self.indx_start]) > self.results_size:
                self.L_runs_results[self.indx_start].pop(0)

            assert len(self.L_runs_results[self.indx_start]) <= self.results_size

            done = True

            reward = self.compute_reward(self.state, self.goal, info)

            # if self.indx_goal < len(self.starting_state_set) :
            #     ## add new sample to the set of starting_states
            #     self.starting_state_set[self.indx_goal].append(new_state)



            ## UPDATE SKIPPING RESULTS
            if self.allow_skipping and self.skipping == True:
                self.skipping_results[self.indx_goal].append(1)

                if len(self.skipping_results[self.indx_goal]) > self.results_size:

                    self.skipping_results[self.indx_goal].pop(0)

                    nb_skip_success = self.skipping_results[self.indx_goal].count(1)
                    nb_skip_fails = self.skipping_results[self.indx_goal].count(0)

                    s_r = float(nb_skip_success/(nb_skip_fails + nb_skip_success))
                    assert s_r >= 0. and s_r <= 1.
                    print("success_ratio skip nb " + str(self.indx_goal) + " = " + str(s_r))

                    if s_r > self.skipping_success_ratio :
                        # remove starting state
                        self.starting_state_set.pop(self.indx_goal-1)
                        self.L_states.pop(self.indx_goal-1)

                        print("self.indx_start = ", self.indx_start)
                        print("self.indx_goal -1  = ", self.indx_goal -1)
                        print("self.L_steps[self.indx_start] = ", self.L_steps[self.indx_start])
                        print("self.L_steps[self.indx_goal-1] = ", self.L_steps[self.indx_goal-1])
                        print("self.L_steps = ", self.L_steps)

                        # remove steps after adding them to the kept starting state
                        self.L_steps[self.indx_start] += self.L_steps[self.indx_goal-1]
                        print("self.L_steps[self.indx_start] = ", self.L_steps[self.indx_start])

                        self.L_steps.pop(self.indx_goal-1)
                        print("self.L_steps = ", self.L_steps)

                        assert sum(self.L_steps) == self.total_steps # make sure we kept enough timesteps

                        # empty results buffer for the successfuly reached goal and the next goal
                        if self.indx_goal + 1 < len(self.skipping_results):
                            self.skipping_results[self.indx_goal+1] = []
                        self.skipping_results[self.indx_goal] = []

                        # remove skipping results for the skipped index
                        self.skipping_results.pop(self.indx_goal-1)

                        # make sure we removed everything properly
                        assert len(self.L_states) == len(self.L_steps) and len(self.L_steps) == len(self.skipping_results)


            #prev_goal = copy.deepcopy(self.goal)
            prev_goal = self.goal_space_projection(self.goal).copy()
            info['done'] = done
            info['goal'] = self.goal_space_projection(self.goal).copy()
            info['traj'] = self.traj
            self.reset()
            return OrderedDict([
                    ("observation", self.state.copy()),
                    ("achieved_goal", self.goal_space_projection(self.state).copy()),
                    ("desired_goal", prev_goal)]), reward, done, info

        elif self.rollout_steps >= self.max_steps:

            self.L_runs_results[self.indx_start].append(0)

            if len(self.L_runs_results[self.indx_start]) > self.results_size:
                self.L_runs_results[self.indx_start].pop(0)

            ## UPDATE SKIPPING RESULTS (NO SKIPPING CONFIRMATION HERE)
            if self.allow_skipping and self.skipping == True:
                print("self.indx_goal = ", self.indx_goal)
                print("len(self.skipping_results) = ", len(self.skipping_results))
                self.skipping_results[self.indx_goal].append(0)

                if len(self.skipping_results[self.indx_goal]) > self.results_size:
                    self.skipping_results[self.indx_goal].pop(0)

            reward = self.compute_reward(self.state, self.goal, info)


            if self.indx_goal < len(self.starting_state_set) :
                ## add new starting_state
                to_add = True

                achieved_goal = new_state[:2].copy()

                for state in self.starting_state_set[self.indx_goal]:

                    goal = np.array(state[:2]).copy()
                    #print("achieved_goal = ", achieved_goal)
                    #print("goal = ", goal)

                    if np.linalg.norm(achieved_goal - goal, axis = -1) < self.width_success:
                        to_add = False

                        break

                if to_add :
                    self.starting_state_set[self.indx_goal].append(new_state)


            done = True
            # prev_goal = copy.deepcopy(self.goal)
            prev_goal = self.goal_space_projection(self.goal).copy()
            info['done'] = done
            info['goal'] = self.goal_space_projection(self.goal).copy()
            info['traj'] = self.traj
            self.reset()

            return OrderedDict([
                    ("observation", self.state.copy()),
                    ("achieved_goal", self.goal_space_projection(self.state).copy()),
                    ("desired_goal", prev_goal)]), reward, done, info

        else:

            done = False

            reward = self.compute_reward(self.state, self.goal, info)

            info['done'] = done
            return OrderedDict([
                    ("observation", self.state.copy()),
                    ("achieved_goal", self.goal_space_projection(self.state).copy()),
                    ("desired_goal", self.goal_space_projection(self.goal).copy()),]), reward, done, info

    def step_test(self, action) :

        obs = self.state_vector()

        for i in range(self.frame_skip):
            new_obs, reward, done, info = self._step(action)
            self.traj.append(new_obs[:2])

        self.rollout_steps += 1

        dst = np.linalg.norm(self.goal_space_projection(self.state) - self.goal_space_projection(self.goal))
        info = {'target_reached': dst<= self.width_test}
        #reward = -dst

        reward = 0.

        return OrderedDict([
                ("observation", self.state.copy()),
                ("achieved_goal", self.goal_space_projection(self.state).copy()),
                ("desired_goal", self.goal_space_projection(self.goal).copy()),]), reward, done, info

    def state_vector(self):
        return self.state

    def _get_obs(self):
        return OrderedDict(
            [
                ("observation", self.state.copy()),
                ("achieved_goal", self.goal_space_projection(self.state).copy()),
                ("desired_goal", self.goal_space_projection(self.goal).copy()),
            ]
        )

    def goal_vector(self):
        return self.goal

    def set_state(self, starting_state):
        self.state = np.array(starting_state)
        return self.state_vector()

    def set_goal_state(self, goal_state):
        self.goal = self.goal_space_projection(np.array(goal_state))
        return 0

    def goal_space_projection(self, obs):
        return obs[:2]

    def sample_task(self):
        """
        Sample task for low-level policy training
        """

        delta_step = 1

        ## make sure the ratio can be computed (at least a success or a failure)
        weights_available = True
        for i in range(0,len(self.L_runs_results)- delta_step ):
            if len(self.L_runs_results[i]) == 0:
                weights_available = False

        # print("weights_available = ", weights_available)

        ## weighted sampling of the starting state
        if self.weighted_selection and weights_available:
            L_weights = []
            for i in range(0, len(self.L_states) - delta_step ):
                # print("self.L_runs_success[i] = ", self.L_runs_success[i])
                # print("self.L_runs_fails[i] = ", self.L_runs_fails[i])

                # print("self.L_runs_results[i] = ", self.L_runs_results[i])

                nb_runs_success = self.L_runs_results[i].count(1)
                nb_runs_fails = self.L_runs_results[i].count(0)

                s_r = float(nb_runs_success/(nb_runs_fails + nb_runs_success))

                ## on cape l'inversion
                if s_r <= 0.1:
                    s_r = 10
                else:
                    s_r = 1./s_r

                L_weights.append(s_r)

            max = sum(L_weights)
            pick = random.uniform(0, max)

            current = 0
            for i in range(0,len(L_weights)):
                s_r = L_weights[i]
                current += s_r
                if current > pick:
                    break

            self.indx_start = i

        # targeted selection
        elif self.target_selection :

            pick = np.random.random()

            if self.model != None :
                if pick < self.target_ratio and self.model.last_failed_task != None:
                    print("\ntarget sampling")
                    self.indx_start = self.model.last_failed_task
                else:
                    self.indx_start = random.randint(0, len(self.L_states) - delta_step - 1 )
            else:
                self.indx_start = random.randint(0, len(self.L_states) - delta_step - 1 )

        ## uniform selection
        else:
            #self.indx_start = random.randint(0, len(self.L_states) - delta_step - 1 )
            self.indx_start = 0


        pick = np.random.random()

        if self.allow_skipping and pick < self.skipping_ratio and self.indx_start  < len(self.L_states) - delta_step - 2:
            delta_step += 1
            self.skipping = True

        self.indx_goal = self.indx_start + delta_step
        length_task = sum(self.L_steps[self.indx_start:self.indx_goal]) #+ 5  ## bonus timesteps pour compenser les starting states reculés.
        # starting_state = self.L_states[self.indx_start]
        #starting_state = random.choice(self.starting_state_set[self.indx_start])
        starting_state = self.starting_state_set[self.indx_start][0]

        if self.indx_start == 1:
            starting_state = random.choice(self.starting_state_set[self.indx_start])
        else:
            starting_state = self.L_states[self.indx_start]

        if self.indx_goal == 1:
            goal_state = random.choice(self.starting_state_set[self.indx_goal])
        else:
            goal_state = self.L_states[self.indx_goal]

        return starting_state, length_task, goal_state


    def reset(self):
        # print("state = ", self.state_vector())
        self.reset_primitive()
        self.testing = False
        self.skipping = False

        starting_state, length_task, goal_state = self.sample_task()

        self.set_goal_state(goal_state)
        self.set_state(starting_state)
        self.max_steps = length_task
        #self.max_steps = 10

        self.rollout_steps = 0
        self.traj = []
        self.first_part_trans = []

        self.traj.append(self.state_vector())

        return OrderedDict([
                ("observation", self.state.copy()),
                ("achieved_goal", self.goal_space_projection(self.state.copy())),
                ("desired_goal", self.goal_space_projection(self.goal.copy())),])


class DubinsMazeEnvGCPHERSB3_deconstructed(DubinsMazeEnv):

    def __init__(self, L_states, L_steps, args={
            'mazesize':5,
            'random_seed':0,
            'mazestandard':False,
            'wallthickness':0.1,
            'wallskill':True,
            'targetkills':True,
            'max_steps': 50,
            'width': 0.1
        }):

        super(DubinsMazeEnvGCPHERSB3_deconstructed,self).__init__(args = args)

        print("state = ", self.state)

        self.max_steps = args['max_steps']
        # Counter of steps per episode
        self.args = args
        self.rollout_steps = 0

        self.action_space = spaces.Box(np.array([0., -1.]), np.array([1., 1.]), dtype = np.float32)


        ms = int(args['mazesize'])
        self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(
                        low = np.array([0,0,-4]),
                        high = np.array([ms,ms,4])),
                    "achieved_goal": spaces.Box(
                        low = np.array([0,0]),
                        high = np.array([ms,ms])),
                    "desired_goal": spaces.Box(
                        low = np.array([0,0]),
                        high = np.array([ms,ms])),
                }
            )

        self.state =  np.array([0.5, 0.5, 0.])
        self.goal= np.array([args['mazesize']-0.5, args['mazesize']-0.5])
        # self.width = args['width']
        self.width_bonus = 0.3
        self.width_success = 0.3
        self.width = self.width_bonus
        self.width_test = self.width_success
        self.L_states = L_states
        self.L_steps = L_steps
        self.total_steps = 0
        self.starting_state_set = []
        self.L_runs_results = []
        self.results_size = 10

        self.success_bonus = 10
        self.traj = []
        self.indx_start = 0
        self.indx_goal = -1
        self.testing = False
        self.expanded = False

        self.buffer_transitions = []

        self.weighted_selection = True

        self.testing_traj = False
        self.fixed_goal_indx = 1

    def state_act(self, action):
        r = np.copy(self.state)
        return self.update_state(r, action, self.delta_t)


    def compute_reward(self, achieved_goal, desired_goal, info):

        if len(achieved_goal.shape) ==  1:
            dst = np.linalg.norm(self.goal_space_projection(achieved_goal) - self.goal_space_projection(desired_goal), axis = -1)
            _info = {'target_reached': dst<= self.width_success}

            if _info['target_reached']:
                return -0.1

            else:
                return -1.

        else:
            ## compute -1 or 0 reward
            distances = np.linalg.norm(achieved_goal[:,:2] - desired_goal[:,:2], axis=-1)  ## pour éviter le cas ou dist == 0.
            distances_mask = (distances <= self.width_bonus).astype(np.float32)
            rewards = distances_mask - 1. - distances_mask * 0.1 # {-1, -0.1}

            return rewards

    def _step_train(self, action) :

        state = self.state_vector()
        new_state, reward, done, info = self._step(action)
        self.traj.append(new_state)

        self.rollout_steps += 1

        dst = np.linalg.norm(self.goal_space_projection(self.state) - self.goal_space_projection(self.goal))
        info = {'target_reached': dst<= self.width_success}

        self.buffer_transitions = [] ## reset to empty list before, eventually adding transitions

        if info['target_reached']: # achieved goal
            done = True
            reward = self.compute_reward(self.state, self.goal, info)

            if self.indx_goal < len(self.starting_state_set) :
                ## add new sample to the set of starting_states
                self.starting_state_set[self.indx_goal].append(new_state)

            prev_goal = copy.deepcopy(self.goal)
            info['done'] = done
            info['traj'] = self.traj
            self.reset()
            return OrderedDict([
                    ("observation", self.state.copy()),
                    ("achieved_goal", self.goal_space_projection(self.state).copy()),
                    ("desired_goal", self.goal_space_projection(prev_goal).copy()),]), reward, done, info

        elif self.rollout_steps >= self.max_steps:

            reward = self.compute_reward(self.state, self.goal, info)
            done = True
            prev_goal = copy.deepcopy(self.goal)
            info['done'] = done
            info['traj'] = self.traj
            self.reset()

            return OrderedDict([
                    ("observation", self.state.copy()),
                    ("achieved_goal", self.goal_space_projection(self.state).copy()),
                    ("desired_goal", self.goal_space_projection(prev_goal).copy()),]), reward, done, info

        else:

            done = False
            reward = self.compute_reward(self.state, self.goal, info)
            info['done'] = done
            return OrderedDict([
                    ("observation", self.state.copy()),
                    ("achieved_goal", self.goal_space_projection(self.state).copy()),
                    ("desired_goal", self.goal_space_projection(self.goal).copy()),]), reward, done, info

    def _step_test(self, action) :

        state = self.state_vector()
        new_state, reward, done, info = self._step(action)
        self.traj.append(new_state)

        self.rollout_steps += 1

        dst = np.linalg.norm(self.goal_space_projection(self.state) - self.goal_space_projection(self.goal))
        info = {'target_reached': dst<= self.width_success}

        self.buffer_transitions = [] ## reset to empty list before, eventually adding transitions

        if info['target_reached']: # achieved goal
            done = False
            reward = self.compute_reward(self.state, self.goal, info)

            if self.indx_goal < len(self.starting_state_set) :
                ## add new sample to the set of starting_states
                self.starting_state_set[self.indx_goal].append(new_state)

            prev_goal = copy.deepcopy(self.goal)
            info['done'] = done
            info['traj'] = self.traj

            ## update goal
            self.indx_goal += 1
            goal_state = self.L_states[self.indx_goal]
            self.set_goal_state(goal_state)

            self.rollout_steps = 0
            self.max_steps = sum(self.L_steps[self.indx_goal-1:self.indx_goal])

            return OrderedDict([
                    ("observation", self.state.copy()),
                    ("achieved_goal", self.goal_space_projection(self.state).copy()),
                    ("desired_goal", self.goal_space_projection(prev_goal).copy()),]), reward, done, info

        elif self.rollout_steps >= self.max_steps:

            reward = self.compute_reward(self.state, self.goal, info)
            done = True
            prev_goal = copy.deepcopy(self.goal)
            info['done'] = done
            info['traj'] = self.traj
            self.reset()

            return OrderedDict([
                    ("observation", self.state.copy()),
                    ("achieved_goal", self.goal_space_projection(self.state).copy()),
                    ("desired_goal", self.goal_space_projection(prev_goal).copy()),]), reward, done, info

        else:

            done = False
            reward = self.compute_reward(self.state, self.goal, info)
            info['done'] = done
            return OrderedDict([
                    ("observation", self.state.copy()),
                    ("achieved_goal", self.goal_space_projection(self.state).copy()),
                    ("desired_goal", self.goal_space_projection(self.goal).copy()),]), reward, done, info

    def step(self, action):
        if self.testing_traj:
            return self._step_test(action)

        else:
            return self._step_train(action)

    def step_test(self, action) :

        obs = self.state_vector()
        new_obs, reward, done, info = self._step(action)
        self.traj.append(new_obs[:2])

        self.rollout_steps += 1

        dst = np.linalg.norm(self.goal_space_projection(self.state) - self.goal_space_projection(self.goal))
        info = {'target_reached': dst<= self.width_test}
        #reward = -dst

        reward = 0.

        return OrderedDict([
                ("observation", self.state.copy()),
                ("achieved_goal", self.goal_space_projection(self.state).copy()),
                ("desired_goal", self.goal_space_projection(self.goal).copy()),]), reward, done, info

    def state_vector(self):
        return self.state

    def _get_obs(self):
        return OrderedDict(
            [
                ("observation", self.state.copy()),
                ("achieved_goal", self.goal_space_projection(self.state).copy()),
                ("desired_goal", self.goal_space_projection(self.goal).copy()),
            ]
        )

    def goal_vector(self):
        return self.goal

    def set_state(self, starting_state):
        self.state = np.array(starting_state)
        return self.state_vector()

    def set_goal_state(self, goal_state):
        self.goal = self.goal_space_projection(np.array(goal_state))
        return 0

    def goal_space_projection(self, obs):
        return obs[:2]

    def sample_task(self):
        """
        Sample task for low-level policy training
        """

        if not self.testing_traj:

            delta_step = 1

            self.indx_start = random.randint(0, len(self.L_states) - delta_step - 1 )
            self.indx_goal = self.indx_start + delta_step
            length_task = sum(self.L_steps[self.indx_start:self.indx_goal]) #+ 5  ## bonus timesteps pour compenser les starting states reculés.
            # starting_state = self.L_states[self.indx_start]
            starting_state = random.choice(self.starting_state_set[self.indx_start])

            goal_state = self.L_states[self.indx_goal]

        else:
            self.indx_start = 0
            self.indx_goal = self.indx_start + 1

            length_task = sum(self.L_steps[self.indx_start:self.indx_goal]) #+ 5  ## bonus timesteps pour compenser les starting states reculés.
            # starting_state = self.L_states[self.indx_start]
            starting_state = random.choice(self.starting_state_set[self.indx_start])

            goal_state = self.L_states[self.indx_goal]

        return starting_state, length_task, goal_state


    def reset(self):
        # print("state = ", self.state_vector())
        self.reset_primitive()
        self.testing = False
        self.skipping = False

        starting_state, length_task, goal_state = self.sample_task()

        self.set_goal_state(goal_state)
        self.set_state(starting_state)
        self.max_steps = length_task
        #self.max_steps = 10

        self.rollout_steps = 0
        self.traj = []
        self.first_part_trans = []

        self.traj.append(self.state_vector())

        return OrderedDict([
                ("observation", self.state.copy()),
                ("achieved_goal", self.goal_space_projection(self.state.copy())),
                ("desired_goal", self.goal_space_projection(self.goal.copy())),])



class DubinsMazeEnvGCPSB3_V2(DubinsMazeEnv):

    def __init__(self, L_states, L_steps, args={
            'mazesize':5,
            'random_seed':0,
            'mazestandard':False,
            'wallthickness':0.1,
            'wallskill':True,
            'targetkills':True,
            'max_steps': 50,
            'width': 0.1
        }):

        #MazeEnv.__init__(self, args = args)
        super(DubinsMazeEnvGCPSB3_V2,self).__init__(args = args)

        print("state = ", self.state)

        self.max_steps = args['max_steps']
        # Counter of steps per episode
        self.args = args
        self.rollout_steps = 0

        self.action_space = spaces.Box(np.array([0., -1.]), np.array([1., 1.]), dtype = np.float32)
        low = np.array([0.,0.,-4., 0., 0., -4])
        high = np.array([args['mazesize'], args['mazesize'],4., args['mazesize'], args['mazesize'], 4.])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.state =  np.array([0.5, 0.5, 0.])
        self.goal= np.array([args['mazesize']-0.5, args['mazesize']-0.5])
        self.width = args['width']
        self.width = 0.2
        self.width_test = 0.2
        self.L_states = L_states
        self.L_steps = L_steps
        self.starting_state_set = []
        self.success_bonus = 10
        self.traj = []
        self.indx_start = 0
        self.indx_goal = -1
        self.testing = False
        self.expanded = False

        self.buffer_transitions = []
        self.go_explore_prob = 0.5


    def state_act(self, action):
        r = np.copy(self.state)
        return self.update_state(r, action, self.delta_t)


    def step(self, action) :

        obs = self.state_vector()
        new_obs, reward, done, info = self._step(action)
        self.traj.append(new_obs[:2])

        self.rollout_steps += 1

        norm_new_obs = copy.deepcopy(new_obs)
        norm_new_obs[0] = 2.*(norm_new_obs[0] - self.observation_space.low[0])/(self.observation_space.high[0] - self.observation_space.low[0]) -1.
        norm_new_obs[1] = 2.*(norm_new_obs[1] - self.observation_space.low[1])/(self.observation_space.high[1] - self.observation_space.low[1]) -1.
        norm_new_obs[2] = 2.*(norm_new_obs[2] - self.observation_space.low[2])/(self.observation_space.high[2] - self.observation_space.low[2]) -1.

        norm_goal = copy.deepcopy(self.goal)
        norm_goal[0] = 2.*(norm_goal[0] - self.observation_space.low[0])/(self.observation_space.high[0] - self.observation_space.low[0]) -1.
        norm_goal[1] = 2.*(norm_goal[1] - self.observation_space.low[1])/(self.observation_space.high[1] - self.observation_space.low[1]) -1.
        norm_goal[2] = 2.*(norm_goal[2] - self.observation_space.low[2])/(self.observation_space.high[2] - self.observation_space.low[2]) -1.

        ## compute 2 different distances
        dst_total = np.linalg.norm(norm_new_obs - norm_goal)
        dst = np.linalg.norm(new_obs[:2] - self.goal[:2])

        info = {'target_reached': dst<= self.width}
        reward = -dst_total

        self.buffer_transitions = [] ## reset to empty list before, eventually adding transitions

        if info['target_reached']:
            done = True

        if info['target_reached']: # achieved goal

            if self.indx_goal < len(self.starting_state_set) :
                ## add new sample to the set of starting_states
                self.starting_state_set[self.indx_goal].append(new_obs)

            if not self.testing and not self.expanded and random.random() < self.go_explore_prob and self.indx_goal < len(self.L_states) - 1:
                ## continue learning with next goal if not testing and next goal available
                print("Go_explore trick")

                self.expanded = True
                info['done'] = True
                info['traj'] = self.traj
                # done = True
                #print("self.goal = ", type(self.goal))
                prev_goal = copy.deepcopy(self.goal) ## save previous goal for transitions
                #print("prev_goal = ", type(prev_goal))

                ## modify goal and extend trajectory for next goal -> no reset()
                self.indx_start += 1
                self.indx_goal += 1
                length_task = sum(self.L_steps[self.indx_start:self.indx_goal]) ## bonus timesteps
                goal_state = self.L_states[self.indx_goal]
                self.set_goal_state(goal_state)
                self.max_steps = length_task
                self.rollout_steps = 0

                norm_goal = copy.deepcopy(self.goal)
                norm_goal[0] = 2.*(norm_goal[0] - self.observation_space.low[0])/(self.observation_space.high[0] - self.observation_space.low[0]) -1.
                norm_goal[1] = 2.*(norm_goal[1] - self.observation_space.low[1])/(self.observation_space.high[1] - self.observation_space.low[1]) -1.
                norm_goal[2] = 2.*(norm_goal[2] - self.observation_space.low[2])/(self.observation_space.high[2] - self.observation_space.low[2]) -1.

                ## compute 2 different distances
                new_dst_total = np.linalg.norm(norm_new_obs - norm_goal)
                new_dst =  np.linalg.norm(new_obs[:2] - self.goal[:2])

                new_info = {'target_reached': new_dst<= self.width}
                new_info['done'] = False
                new_done = False
                new_reward = -new_dst_total

                self.buffer_transitions.append((np.hstack((obs, self.goal)), action, np.hstack((new_obs, self.goal)), new_reward, new_done, new_info))

                return np.hstack((new_obs, prev_goal)), reward, done, info

            else:
                prev_goal = copy.deepcopy(self.goal)
                info['done'] = done
                info['traj'] = self.traj
                self.reset()
                return np.hstack((new_obs, prev_goal)), reward, done, info

        elif self.rollout_steps >= self.max_steps:
            done = True
            prev_goal = copy.deepcopy(self.goal)
            info['done'] = done
            info['traj'] = self.traj
            self.reset()
            return np.hstack((new_obs, prev_goal)), reward, done, info

        else:
            info['done'] = done
            #info['traj'] = self.traj
            return np.hstack((new_obs, self.goal)), reward, done, info

    def step_test(self, action) :

        obs = self.state_vector()
        new_obs, reward, done, info = self._step(action)
        self.traj.append(new_obs[:2])

        self.rollout_steps += 1

        ## one single distance needed to get the target_reached boolean
        dst = np.linalg.norm(self.state[:2] - self.goal[:2])
        info = {'target_reached': dst<= self.width_test}
        reward = -dst

        return np.hstack((new_obs, self.goal)), reward, done, info

    def state_vector(self):
        return self.state

    def goal_vector(self):
        return self.goal

    def set_state(self, starting_state):
        self.state = np.array(starting_state)
        return self.state_vector()

    def set_goal_state(self, goal_state):
        self.goal = np.array(goal_state)
        return 0

    def sample_task(self):
        """
        Sample task for low-level policy training
        """

        delta_step = 1
        self.indx_start = random.randint(0, len(self.L_states) - delta_step -1)
        self.indx_goal = self.indx_start + delta_step
        length_task = sum(self.L_steps[self.indx_start:self.indx_goal])

        # starting_state = self.L_states[self.indx_start]
        starting_state = random.choice(self.starting_state_set[self.indx_start])

        goal_state = self.L_states[self.indx_goal]

        return starting_state, length_task, goal_state

    def reset(self):
        # print("state = ", self.state_vector())
        self.reset_primitive()
        self.testing = False
        self.expanded = False

        starting_state, length_task, goal_state = self.sample_task()

        self.set_goal_state(goal_state)
        self.set_state(starting_state)
        self.max_steps = length_task

        self.rollout_steps = 0
        self.traj = []
        self.traj.append(self.state_vector()[:2])

        #print("reset = ", np.hstack((self.state_vector(), self.goal)))

        return np.hstack((self.state_vector(), self.goal))


class DubinsMazeEnv_BP_SB3(DubinsMazeEnv):

    def __init__(self, demo_traj, args={
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
                info['done'] = done
                return new_obs, 1., done, info

        else:
            if done or self.rollout_steps >= self.max_steps:
                done = True
                info['done'] = done
                info['traj'] = self.traj
                self.reset()
                return new_obs, 0., done, info
            else:
                info['done'] = done
                return new_obs, 0., done, info

    def set_state(self, state):
        self.state = state
        return self.state_vector()

    def set_starting_indx(self, starting_indx):
        self.starting_indx = starting_indx
        return 0

    def set_task_length(self, task_length):
        self.max_steps = task_length
        return 0

    def set_goal_state(self, goal_state):
        self.goal = goal_state
        return 0

    def reset(self):
        # print("state = ", self.state_vector())
        self.reset_primitive()
        self.set_state(self.demo_traj[self.starting_indx])
        self.task_length = len(self.demo_traj) - self.starting_indx
        self.rollout_steps = 0
        self.traj = []
        self.traj.append(self.state_vector())

        return self.state_vector()

# class MazeEnv_BPCS_SB3(MazeEnv):
#
#     def __init__(self, demo_traj, args={
#             'mazesize':15,
#             'random_seed':0,
#             'mazestandard':False,
#             'wallthickness':0.1,
#             'wallskill':True,
#             'targetkills':True,
#             'max_steps': 15
#         }):
#
#         #MazeEnv.__init__(self, args = args)
#         super(MazeEnv_BPCS_SB3,self).__init__(args = args)
#
#         print("MazeEnv.state = ", self.state)
#
#         self.action_space = spaces.Box(np.array([-0.45,-0.45]),np.array([0.45,0.45]))
#         self.wallskill = False
#         self.max_steps = args['max_steps']
#         # Counter of steps per episode
#         self.rollout_steps = 0
#         self.demo_traj = demo_traj
#         self.starting_indx = 0
#         self.width = 0.5
#         self.traj = []
#
#     def step(self, action) :
#
#         obs = self.state_vector()
#         new_obs, reward, done, info = self._step(action)
#         self.traj.append(new_obs)
#
#         #print("reward = ", reward)
#
#         self.rollout_steps += 1
#
#         dst = np.linalg.norm(self.state - self.goal)
#         info = {'target_reached': dst<= self.width}
#         reward = -dst
#
#         if info['target_reached']:
#             done = True
#
#         if info['target_reached']: # achieved goal
#             info['done'] = done
#             info['traj'] = self.traj
#             self.reset()
#             return new_obs, reward, done, info
#
#         elif self.rollout_steps >= self.max_steps:
#             done = True
#             info['done'] = done
#             info['traj'] = self.traj
#             self.reset()
#             return new_obs, reward, done, info
#
#         else:
#             info['done'] = done
#             info['traj'] = self.traj
#             return new_obs, reward, done, info
#
#
#     def set_state(self, state):
#         self.state = state
#         return self.state_vector()
#
#     def state_vector(self):
#         return np.array(self.state)
#
#     def goal_vector(self):
#         return np.array(self.goal)
#
#     def set_starting_indx(self, starting_indx):
#         self.starting_indx = starting_indx
#         return 0
#
#     def set_task_length(self, task_length):
#         self.max_steps = task_length
#         return 0
#
#     def set_goal_state(self, goal_state):
#         self.goal = goal_state
#         return 0
#
#     def set_goal_indx(self, goal_indx):
#         self.goal = self.demo_traj[goal_indx]
#         return 0
#
#     def reset(self):
#         # print("state = ", self.state_vector())
#         self.reset_primitive()
#         new_state = np.array(self.demo_traj[self.starting_indx])
#         self.set_state(new_state)
#         self.task_length = len(self.demo_traj) - self.starting_indx
#         self.rollout_steps = 0
#         self.traj = []
#         self.traj.append(self.state_vector())
#
#         return self.state_vector()
