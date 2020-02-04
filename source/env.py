# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adheesh
# author Bala Murali
# Copyright
# brief Environment class for the simulation

import numpy as np
import random
import copy

from constants import CONST
from agent import Agent
from mobile_robot import MobileRobot
from render import Render

np.set_printoptions(precision=3, suppress=True)
class Env:
    def __init__(self, numDrones, numMobileRobs):
        self.agents = self.initAgents(CONST.NUM_DRONES)
        self.timeStep = CONST.TIME_STEP
        
        
    def initTotalArea(self):
        # beyond = 0
        # unexplored = 50
        # explored = 255
        # drone pos = 100
        
        return 0
        
    def initAgents(self, n):
        agents = []
        for i in range(0,n):
            agents.append(Agent())
        return agents
      
    def reset(self):
        self.agents = self.initAgents(len(self.agents))
        # need to update initial state for reset function
        initialState = 0
        return initialState
        
    def getActionSpace(self):
        return [0,1,2,3,4]
    
    def getStateSpace(self):
        
        return 0
    
    def stepAgent(self, actions):
        # have to decide on the action space
        # waypoints or velocity
        posOut = []
        velOut = []
        for agent, action in zip(self.agents, actions):
            vel = np.array([0,0])
            if action == 0:
                pass
            elif action == 1:
                vel[1] = 1
            elif action == 2:
                vel[0] = -1
            elif action == 3:
                vel[1] = -1
            elif action == 4:
                vel[0] = 1
            agent.setParams(vel)
            agent.updateState(self.timeStep)
            curState = agent.getState()
            posOut.append(curState[0])
            velOut.append(curState[1])
        return posOut, velOut
    
    def step(self, agentActions):
        dronePos, droneVel = self.stepDrones(agentActions)
        # update reward mechanism
        reward = self.getReward()
        return dronePos, reward
                

    def render(self):
        pass
    
            
    def updateArea(self):
        pass 
        
    def getReward(self):
            reward = 0
            return reward
                    


