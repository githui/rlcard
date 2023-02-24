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

trainer = DMCTrainer(
    env,
    cuda=args.cuda,
    load_model=args.load_model,
    xpid=args.xpid,
    savedir=args.savedir,
    save_interval=args.save_interval,
    num_actor_devices=args.num_actor_devices,
    num_actors=args.num_actors,
    training_device=args.training_device,
)

# Train DMC Agents
trainer.start()