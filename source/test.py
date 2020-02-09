# -*- coding: utf-8 -*-

import numpy as np
from env import Env
from tqdm import tqdm
import cv2
import random as rand

np.set_printoptions(threshold = np.inf, linewidth = 1000 ,precision=3, suppress=True)

env = Env()

NUM_EPISODES = 100
LEN_EPISODES = 200

act = 0 
wait = True

for episode in tqdm(range(NUM_EPISODES)):
    for step in range(LEN_EPISODES):
        # step agent
        actions = [act]
        agentPosList, rewardList = env.step(actions)
#        print(agentPosList)
        env.render()
        wait = True
        while(wait):
            k = cv2.waitKeyEx(1) 
#            print(k)
            if k == 2490368:
                act = 1
                wait = False
            elif k == 2424832:
                act = 2
                wait = False
            elif k == 2621440:
                act = 3
                wait = False
            elif k == 2555904:
                act = 4
                wait = False
            