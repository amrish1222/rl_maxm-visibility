# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 19:23:19 2020

@author: amris
"""

from env import Env
from tqdm import tqdm
import cv2
env = Env()

NUM_EPISODES = 100
LEN_EPISODES = 200

for episode in tqdm(range(NUM_EPISODES)):
    for step in range(LEN_EPISODES):
        # step agent
        actions = [1]
        agentPosList, rewardList = env.step(actions)
        print(agentPosList)
        cv2.waitKey(0)