# from .mazeenv.mazeenv import MazeEnv
import gym
from gym.envs.registration import register

print("REGISTERING MazeEnv5-v0")
register(
    id='MazeEnv5-v0',
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
