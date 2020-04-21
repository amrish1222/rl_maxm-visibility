# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
import keyboard

import numpy as np
from env import Env
from tqdm import tqdm
import cv2
import random as rand
import matplotlib.pyplot as plt
from collections import defaultdict

import SimpleNNagent as sNN
import simpleCNNagent as cNN
from constants import CONSTANTS
CONST = CONSTANTS()

from time import time as tm

np.set_printoptions(threshold = np.inf, linewidth = 1000 ,precision=3, suppress=True)

def getKeyPressOld(act):
    k = cv2.waitKeyEx(1) 
    #            print(k)
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


NUM_EPISODES = 50000
LEN_EPISODES = 125
curState = []
newState= []
reward_history = []
mapNewVisPenalty_history = defaultdict(list)
loss_history = []
totalViewed = []
dispFlag = False

curRawState = env.reset()
curState = rlAgent.formatInput(curRawState)
#rlAgent.summaryWriter_showNetwork(curState[0])

keyPress = 1

for episode in tqdm(range(NUM_EPISODES)):
#    LEN_EPISODES = 25 + min(int(episode* 5 /50),100)
    curRawState = env.reset()
    
    # generate state for each agent
    curState = rlAgent.formatInput(curRawState)
    
    episodeReward  = 0
    epidoseLoss = 0
    episodeNewVisited = 0
    episodePenalty = 0
    
    for step in range(LEN_EPISODES):
        # render environment after taking a step
        keyPress = getKeyPress(keyPress)
        
        if keyPress == 1 and step%4 == 0:
            env.render()
        
        if episode%500 in range(10,15) and step%4 == 0:
            env.save2Vid()
            
        # Get agent actions
        aActions = []
        for i in range(CONST.NUM_AGENTS):
            # get action for each agent using its current state from the network
            aActions.append(rlAgent.EpsilonGreedyPolicy(curState[i]))
        
        # do actions
#        a = tm()
        agentPosList, display, reward, newAreaVis, penalty, done = env.step(aActions)
#        b = tm()
#        print("step: ", round(1000*(b-a), 3))
        # update nextState
        newRawState = []
        for agentPos in agentPosList:
            newRawState.append([agentPos, display])
        newState = rlAgent.formatInput(newRawState)
        
        
        # add to replay memory
        rlAgent.buildReplayMemory(curState[0], newState[0], aActions[0], done, reward)
        
        # train network
        loss = 0
        if len(rlAgent.curState) > rlAgent.batchSize:
            
            # creating the mini batch for training
            loss = rlAgent.buildMiniBatchTrainData()
            
            #training using the mini batches
            rlAgent.trainModel()
            
        # record history
        episodeReward += reward
        epidoseLoss += loss
        episodeNewVisited += newAreaVis
        episodePenalty += penalty
        
        # set current state for next step
        curState = newState
        
        if done:
            break
        
    # post episode
    
    # Epsilon Decay
    if rlAgent.epsilon >= rlAgent.minEpsilon:
        rlAgent.epsilon *= rlAgent.epsilonDecay
#        rlAgent.my_lr_scheduler.step()
    
    
    # Record history        
    reward_history.append(episodeReward)
    loss_history.append(epidoseLoss)
    mapNewVisPenalty_history[env.mapId].append((episodeReward,episodeNewVisited,episodePenalty))
#    totalViewed.append(np.count_nonzero(display==255))
    
    # You may want to plot periodically instead of after every episode
    # Otherwise, things will slow
    rlAgent.summaryWriter_addMetrics(episode, epidoseLoss, reward_history, mapNewVisPenalty_history, LEN_EPISODES)
    if episode % 50 == 0:
        rlAgent.saveModel("checkpoints")
            
    
rlAgent.saveModel("checkpoints")
env.out.release()
        
            