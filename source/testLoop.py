# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
import time
import keyboard

import numpy as np
from env import Env
from tqdm import tqdm
import cv2
import random as rand
import matplotlib.pyplot as plt

import SimpleNNagent as sNN
import simpleCNNagent as cNN
from constants import CONSTANTS
CONST = CONSTANTS()

np.set_printoptions(threshold = np.inf, linewidth = 1000 ,precision=3, suppress=True)

def waitKeyPress():
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
    return act

def getKeyPressOld(act):
    k = cv2.waitKeyEx(1) 
#    print(k)
    if k == 2490368:
        act = 1
    elif k == 2424832:
        act = 2
    elif k == 2621440:
        act = 3
    elif k == 2555904:
        act = 4
    return act

def getKeyPress(act):
    if keyboard.is_pressed('['):
        act = 1
    elif keyboard.is_pressed(']'):
        act = 2
    return act


env = Env()

#rlAgent = sNN.SimpleNNagent(env)
rlAgent = cNN.SimplecNNagent(env)

rlAgent.loadModel("checkpoints/testCNNmultiMap.pt")

NUM_EPISODES = 3000
LEN_EPISODES = 100
curState = []
newState= []
reward_history = []
reward_last100 = []
loss_history = []
dispFlag = True

keyPress = 1
a = time.time()

for episode in tqdm(range(NUM_EPISODES)):
#    LEN_EPISODES = 25 + min(int(episode* 2 /50),80)
    a = time.time()
    curRawState = env.reset()
    b = time.time()
#    print(["reset time = ", round(1000*(b-a),0)])
    
    # generate state for each agent
    curState = rlAgent.formatInput(curRawState)
    
    episodeReward  = 0
    epidoseLoss = 0
    
    for step in range(LEN_EPISODES):
        times = []
        a = time.time()
        # render environment after taking a step
        keyPress = getKeyPress(keyPress)
        if keyPress == 1:
            env.render()
            
        # Get agent actions
        aActions = []
        for i in range(CONST.NUM_AGENTS):
            # get action for each agent using its current state from the network
            aActions.append(rlAgent.getMaxAction(curState[i]))
        
        # do actions
        a = time.time()
        agentPosList, display, reward, done = env.step(aActions)
        b = time.time()
        times.append(["step time", round(1000*(b-a),0)])
        a = time.time()
        # update nextState
        newRawState = []
        for agentPos in agentPosList:
            newRawState.append([agentPos, display])
        newState = rlAgent.formatInput(newRawState)
        
#        print(times)
        
        # record history
#        reward = sum(rewardList)
        episodeReward += reward
        
        # set current state for next step
        curState = newState
        
        
        if done:
            break
        
    # post episode
    
    
    # Record history
    if len(reward_history) < 1:
        reward_last100.append(0)
    elif len(reward_history) <=100:
        reward_last100.append(sum(reward_history)/ len(reward_history))
    else:
        reward_last100.append(sum(reward_history[-100:])/100)
    reward_history.append(episodeReward)
    rlAgent.summaryWriter_addMetrics(episode, 0, episodeReward, reward_last100[-1],step + 1)
    # You may want to plot periodically instead of after every episode
    # Otherwise, things will slow
    if episode % 10 == 0:
        if dispFlag:
            fig = plt.figure(2)
            plt.clf()
            plt.xlim([0,NUM_EPISODES])
            plt.plot(reward_history,'ro')
            plt.plot(reward_last100)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Reward Per Episode')
            plt.pause(0.01)
            fig.canvas.draw()
            
        
            