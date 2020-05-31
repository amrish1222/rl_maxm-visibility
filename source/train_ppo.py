# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
import keyboard

import numpy as np
from env import Env
from tqdm import tqdm
import cv2
from collections import defaultdict
from collections import deque

from ppo_agent import PPO
from ppo_agent import Memory
from constants import CONSTANTS
CONST = CONSTANTS()

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
#    if keyboard.is_pressed('['):
#        act = 1
#    elif keyboard.is_pressed(']'):
#        act = 2
    return act


env = Env()

memory = Memory()
rlAgent = PPO(env)


NUM_EPISODES = 50000
LEN_EPISODES = 500
UPDATE_TIMESTEP = 1000
curState = []
newState= []
reward_history = []
mapNewVisPenalty_history = defaultdict(list)
totalViewed = []
dispFlag = False

#curRawState = env.reset()
#curState = rlAgent.formatInput(curRawState)
#rlAgent.summaryWriter_showNetwork(curState[0])

keyPress = 0
timestep = 0
loss = None

#initialization of history
stateHistQ = deque([])


for episode in tqdm(range(NUM_EPISODES)):
    curRawState = env.reset()
    
    # generate state for each agent
    curState = rlAgent.formatInput(curRawState)
    
    episodeReward  = 0
    epidoseLoss = 0
    episodeNewVisited = 0
    episodePenalty = 0
    
    stateHistQ.clear()
    for i in range(3):
        stateHistQ.append(np.zeros_like(curState[0][0][0]))
    
    for step in range(LEN_EPISODES):
        timestep += 1
        
        # render environment after taking a step
        keyPress = getKeyPress(keyPress)
        
        if keyPress == 1:
            env.render()
        
        if episode%500 in range(10,15) and step%4 == 0:
            env.save2Vid()
            
        
        # append History
        stateHist = np.array(stateHistQ)[0:2]
        stateHist.reshape(2,50,50)
        curState[0][0] = np.concatenate((curState[0][0], stateHist), axis = 0)
        # Get agent actions
        aActions = []
        for i in range(CONST.NUM_AGENTS):
            # get action for each agent using its current state from the network
            action = rlAgent.policy_old.act(curState[i], memory)
            aActions.append(action)
        
        # do actions
        agentPosList, advrsyPosList, display, reward, newAreaVis, penalty, done = env.step(aActions)
        if step == LEN_EPISODES -1:
            done = True
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        
        # update nextState
        newRawState = []
        for agentPos in agentPosList:
            newRawState.append([agentPos, advrsyPosList, display])
        newState = rlAgent.formatInput(newRawState)
        
        stateHistQ.popleft()
        stateHistQ.append(newState[0][0][0])
        
        if timestep % UPDATE_TIMESTEP == 0:
            loss = rlAgent.update(memory)
            memory.clear_memory()
            timestep = 0
        
        # record history
        episodeReward += reward
        episodeNewVisited += newAreaVis
        episodePenalty += penalty
        
        # set current state for next step
        curState = newState
        
        if done:
            break
        
    # post episode
    
    # Record history        
    reward_history.append(episodeReward)
    totalViewed.append(np.count_nonzero(env.currentMapState==255))
    
    # scaling both positive reward and negative reward to [0,1] and adding
    scaledTotalReward = episodeNewVisited / env.numOpenCells + episodePenalty/ (CONST.VISIBILITY_PENALTY* LEN_EPISODES)
    
    mapNewVisPenalty_history[env.mapId].append((scaledTotalReward,
                                                episodeNewVisited / env.numOpenCells,
                                                episodePenalty/ (CONST.VISIBILITY_PENALTY* LEN_EPISODES),
                                                totalViewed[-1]/ env.numOpenCells
                                                ))
    
    # You may want to plot periodically instead of after every episode
    # Otherwise, things will slow
    rlAgent.summaryWriter_addMetrics(episode, loss, reward_history, mapNewVisPenalty_history, LEN_EPISODES)
    if episode % 50 == 0:
        rlAgent.saveModel("checkpoints")
            
    
rlAgent.saveModel("checkpoints")
env.out.release()
        
            