# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:03:40 2023

@author: SUNH
"""

import os
import argparse

import torch
import numpy as np

import rlcard
from rlcard.agents.dmc_agent import DMCTrainer
# from rlcard.games.findfriends import FindfriendsPlayer as Player
from rlcard.games.findfriends.player import FindfriendsPlayer as Playe
from rlcard.agents.dqn_agent import DQNAgent
np_random = np.random.RandomState()
env=rlcard.make('findfriends')
agent=DQNAgent()
env.set_agents(agent)
env.run()