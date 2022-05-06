"""
Environment for SMDP experiments.

How to make the environment:
env = gym.make('MiniGrid-Empty-8x8-v0')
env = TabularWrapper(env)
env = BinaryRewardWrapper(env)
env = WithOptions(env)
_ = env.reset()
s, r, term, info = env.step(action)

s['tabular'] -> is the tabular state
s['image'] -> is the pixel image for deep RL
"""

from gym_minigrid.minigrid import *
from gym import spaces
import gym
import random

MAX_ENV_STEPS = 50

# Online setting
ONLINE_SETTING = True
PENALTY_REWARD = -1
GOAL_REWARD = 10

# Offline setting
# ONLINE_SETTING = False
# PENALTY_REWARD = -3
# GOAL_REWARD = 10


class FlatObsWrapper(gym.core.ObservationWrapper):
    """Fully observable gridworld returning a flat grid encoding."""

    def __init__(self, env):
        super().__init__(env)

        # Since the outer walls are always present, we remove left, right, top, bottom walls
        # from the observation space of the agent. There are 3 channels, but for simplicity
        # in this assignment, we will deal with flattened version of state.

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=((self.env.width - 2) * (self.env.height - 2) * 3,),  # number of cells
            dtype='uint8'
        )
        self.unwrapped.max_steps = MAX_ENV_STEPS

    def observation(self, obs):
        # this method is called in the step() function to get the observation
        # we provide code that gets the grid state and places the agent in it
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])
        full_grid = full_grid[1:-1, 1:-1]  # remove outer walls of the environment (for efficiency)

        flattened_grid = full_grid.ravel()

        tabular = (self.agent_pos[0], self.agent_pos[1], self.agent_dir)

        obs = {
            'image': flattened_grid,
            'tabular': tabular,
        }
        return obs

    def render(self, *args, **kwargs):
        """This removes the default visualization of the partially observable field of view."""
        kwargs['highlight'] = False
        return self.unwrapped.render(*args, **kwargs)


class TabularWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to get a both tabular and image observation. Tabular gets (position_x, position_y, direction).
    """

    class Actions(IntEnum):
        # Turn left, turn right, move forward, do nothing
        left = 0
        right = 1
        forward = 2
        noop = 3

    def __init__(self, env):
        super().__init__(env)
        self.actions = self.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

    def observation(self, obs):
        tabular = (self.agent_pos[0], self.agent_pos[1], self.agent_dir)
        obs = {
            'image': obs['image'],
            'tabular': tabular,
        }
        return obs


class BinaryRewardWrapper(gym.core.RewardWrapper):
    """
    Wrapper to remove discounting from the original reward.
    """

    def __init__(self, env):
        super().__init__(env)

    def reward(self, r):
        if r > 0:
            return GOAL_REWARD
        else:
            return 0


class WithOptions(gym.core.Wrapper):
    """
    Option Wrapper
    """

    class Options(IntEnum):
        # Turn left, turn right, move forward, do nothing
        left = 0
        right = 1
        nochange = 2

    def __init__(self, env):
        super().__init__(env)
        self.options = self.Options
        self.option_step_count = 0
        self.stoch_env = False
        self.cur_pos = None

        self.action_space = spaces.Discrete(len(self.options))

    def step(self, option, k_fwd=None):
        self.option_step_count += 1
        rewards = []

        if option == self.options.left:
            obs, reward, done, info = self.env.step(self.actions.left)
        elif option == self.options.right:
            obs, reward, done, info = self.env.step(self.actions.right)
        else:
            obs, reward, done, info = self.env.step(self.actions.noop)

        self.cur_pos = obs["tabular"]
        rewards.append(reward)

        if k_fwd is None:
            if obs["tabular"][1] == 1 and ONLINE_SETTING:
                k_fwd = 1
            elif (obs["tabular"][-1] == 1) and (obs["tabular"][0] < 2):
                k_fwd = 4
            else:
                k_fwd = 2

        for i in range(k_fwd):
            if not done:
                if self.stoch_env:
                    probs = [self.actions.forward] * 5 + [self.actions.noop] * 2 + [self.actions.left,
                                                                                    self.actions.right]
                    action = random.choice(probs)
                else:
                    action = self.actions.forward
                obs, reward, done, info = self.env.step(action)
                # If we cross row 3, obtain penalty
                if (self.cur_pos[1] < 3) and (3 <= obs["tabular"][1]) and (obs["tabular"][0] != 6):
                    reward = PENALTY_REWARD
                self.cur_pos = obs["tabular"]
                rewards.append(reward)

        self.cur_pos = obs["tabular"]
        return obs, rewards, done, info

    def reset(self, **kwargs):
        self.option_step_count = 0
        self.cur_pos = (1, 1, 0)
        return self.env.reset(**kwargs)
