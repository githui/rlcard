# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:03:40 2023

@author: SUNH
"""

import os
import argparse

import torch

import rlcard
from rlcard.agents.dmc_agent import DMCTrainer

env=rlcard.make('findfriends')

env.run()