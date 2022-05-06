"""
Utilities
"""

import logging
import pandas as pd
from collections import namedtuple, deque
import random
import os
import numpy as np
import pickle
import collections
import itertools

logger = logging.getLogger(__name__)


def human_evaluation(env, agent, human_trajectories, use_soc_state=True):
    rewards = []
    for ep, trajectory in enumerate(human_trajectories):
        env.reset()
        agent.reset()
        for action in trajectory:
            env.act(action)
        terminal = False
        agent_reward = 0  # NOT  including reward accumulated along human trajectory
        s = env.get_soc_state() if use_soc_state else env.get_pixel_state()
        while not terminal:
            action = agent.get_action(s, evaluate=True)
            pixel_state, r, terminal, soc_state = env.act(action)
            s = soc_state if use_soc_state else pixel_state
            agent_reward += r
        rewards.append(agent_reward)
    return rewards


def plot(data={}, loc="visualization.pdf", x_label="", y_label="", title="", kind='line',
         legend=True, index_col=None, clip=None, moving_average=False):
    pass


def write_to_csv(data={}, loc="data.csv"):
    if all([len(data[key]) > 1 for key in data]):
        df = pd.DataFrame(data=data)
        df.to_csv(loc)


class Font:
    purple = '\033[95m'
    cyan = '\033[96m'
    darkcyan = '\033[36m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    bgblue = '\033[44m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'


Transition = namedtuple('Transition', ('s', 'a', 'r', 's2', 'term', 'k'))


class ExperienceReplay(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.size = 0
        self.capacity = capacity

    def add(self, **args):
        self.memory.append(Transition(**args))
        self.size += 1

    def get_head(self):
        """
        Read head of deque and move node to the end of queue after reading
        """
        batch = self.memory.popleft()
        # batch = Transition(*zip(*transition))
        self.size -= 1
        self.add(s=batch.s, a=batch.a,r=batch.r, s2=batch.s2, term=batch.term, k=batch.k)
        return batch.s, batch.a, batch.r, batch.s2, batch.term, batch.k

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            raise ValueError("More data is requested than stored.")
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch.s, batch.a, batch.r, batch.s2, batch.term, batch.k

    def save(self, save_folder):
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        pickle.dump(self.memory, open(save_folder + "/buffer.pkl", 'wb'))

    def load(self, save_folder, size=-1):
        self.memory = pickle.load(open(save_folder + "/buffer.pkl", 'rb'))
        # if size > 0:
        #     self.memory = collections.deque(itertools.islice(self.memory, 0, size))
        if size > 0:
            self.memory = deque(random.sample(self.memory, size))
        self.size = size
        self.capacity = len(self.memory)
        print(f"Replay Buffer loaded with {len(self.memory)} transitions.")

