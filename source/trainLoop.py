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
rlAgent = sNN.SimpleNNagent(env)
NUM_EPISODES = 6000
LEN_EPISODES = 25
curState = []
newState= []
reward_history = []
reward_last100 = []
loss_history = []
totalViewed = []
dispFlag = True

curRawState = env.reset()
curState = rlAgent.formatInput(curRawState)
rlAgent.summaryWriter_showNetwork(curState[0])

keyPress = 1
a = time.time()

for episode in tqdm(range(NUM_EPISODES)):
    LEN_EPISODES = 25 + min(int(episode* 5 /50),125)
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
        # render environment after taking a step
        keyPress = getKeyPress(keyPress)
        
        if keyPress == 1:
            env.render()
            
            
        # Get agent actions
        aActions = []
        for i in range(CONST.NUM_AGENTS):
            # get action for each agent using its current state from the network
            aActions.append(rlAgent.EpsilonGreedyPolicy(curState[i]))
        
        # do actions
        a = time.time()
        agentPosList, display, reward, done = env.step(aActions)
        b = time.time()
        times.append(["Step", round(1000*(b-a),0)])
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
            a = time.time()
            
            loss = rlAgent.buildMiniBatchTrainData()
           
            b = time.time()
            times.append(["buidlBatch", round(1000*(b-a),0)])
            
            a = time.time()
            
            rlAgent.trainModel()
            
            b = time.time()
            times.append(["Train", round(1000*(b-a),0)])
        
#        print(times)
        
        # record history
#        reward = sum(rewardList)
        episodeReward += reward
        epidoseLoss += loss
        
        # set current state for next step
        
        curState = newState
        
#        print("pos", newRawState[0][0])
#        print("action", aActions)
        
        if done:
            break
        
    # post episode
    
    # Epsilon Decay
    if rlAgent.epsilon >= rlAgent.minEpsilon:
        rlAgent.epsilon *= rlAgent.epsilonDecay
#        rlAgent.my_lr_scheduler.step()
    
    
    # Record history
    if len(reward_history) <=100:
        reward_last100.append(0)
    else:
        reward_last100.append(sum(reward_history[-100:])/100)
    reward_history.append(episodeReward)
    loss_history.append(epidoseLoss)
    totalViewed.append(np.count_nonzero(display==255))
    
    # You may want to plot periodically instead of after every episode
    # Otherwise, things will slow
    rlAgent.summaryWriter_addMetrics(episode, epidoseLoss, episodeReward, LEN_EPISODES)
    if episode % 50 == 0:
        if dispFlag:
            fig = plt.figure(2)
            plt.clf()
            plt.xlim([0,NUM_EPISODES])
            plt.plot(reward_history,'ro')
            plt.plot(totalViewed,'g.')
            plt.plot(reward_last100)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Reward Per Episode')
            plt.pause(0.01)
            fig.canvas.draw()
            
            fig = plt.figure(3)
            plt.clf()
            plt.xlim([0,NUM_EPISODES])
            plt.plot(loss_history,'b.')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.title('Loss per episode')
            plt.pause(0.01)
            fig.canvas.draw()
        rlAgent.saveModel("checkpoints")
            
    
rlAgent.saveModel("checkpoints")
        
            