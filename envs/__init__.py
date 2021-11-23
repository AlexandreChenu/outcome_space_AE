# from .mazeenv.mazeenv import MazeEnv
import gym
from gym.envs.registration import register

print("REGISTERING MazeEnv-v0")
register(
    id='MazeEnv-v0',
    entry_point='envs.mazeenv.mazeenv:MazeEnv',
    kwargs={'args': {
            'mazesize':5,
            'random_seed':0,
            'mazestandard':False,
            'wallthickness':0.1,
            'wallskill':True,
            'targetkills':True
        }}
)


print("REGISTERING DubinsMazeEnv-v0")
register(
    id='DubinsMazeEnv-v0',
    entry_point='envs.dubins_mazeenv.mazeenv:DubinsMazeEnv',
    kwargs={'args': {
            'mazesize':5,
            'random_seed':0,
            'mazestandard':False,
            'wallthickness':0.1,
            'wallskill':True,
            'targetkills':True
        }}
)
