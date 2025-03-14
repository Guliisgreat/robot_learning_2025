from hw8.roble.envs import ant
from hw8.roble.envs import cheetah
from hw8.roble.envs import obstacles
from hw8.roble.envs import reacher

from gym.envs.registration import register

def register_envs():
    register(
        id='cheetah-roble-v0',
        entry_point='roble.envs.cheetah:HalfCheetahEnv',
        max_episode_steps=1000,
    )
    register(
        id='ant-roble-v0',
        entry_point='roble.envs.ant:AntEnv',
        max_episode_steps=1000,
    )
    register(
        id='obstacles-roble-v0',
        entry_point='roble.envs.obstacles:Obstacles',
        max_episode_steps=500,
    )
    register(
        id='reacher-roble-v0',
        entry_point='roble.envs.reacher:Reacher7DOFEnv',
        max_episode_steps=500,
    )