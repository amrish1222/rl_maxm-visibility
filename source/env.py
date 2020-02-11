# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adheesh
# author Bala Murali
# Copyright
# brief Environment class for the simulation

import numpy as np
import random
import copy
import math
import cv2

from constants import CONSTANTS as K
CONST = K()
from agent import Agent
from obstacle import Obstacle
obsMap = Obstacle()

from visibility import Visibility
vsb  = Visibility(CONST.MAP_SIZE, CONST.MAP_SIZE)

np.set_printoptions(precision=3, suppress=True)
class Env:
    def __init__(self):
        self.agents = self.initAgents(CONST.NUM_AGENTS)
        self.timeStep = CONST.TIME_STEP
        self.obstacleMap,self.obsPlusViewed, self.currentMapState = self.initTotalArea()
        self.prevUnviewedCount = np.count_nonzero(self.currentMapState==0)
        
    def initTotalArea(self):
        # unviewed = 0
        # viewed = 255
        # obstacle = 150
        # agent Pos = 100
        
        obstacleMap = np.zeros((50,50)) # update with new obstacles
        obstacleMap, arrangement = obsMap.getObstacleMap(obstacleMap, vsb)
        obstacleViewedMap = copy.deepcopy(obstacleMap)
        for agent in self.agents:
            obstacleViewedMap = vsb.updateVsbPolyOnImg([agent.getState()[0]],obstacleViewedMap)
        
        agentPos = [agent.getState()[0] for agent in self.agents]
        gPos = self.cartesian2Grid(agentPos)
        currentMapState = self.updatePosMap(gPos, obstacleViewedMap)
        return obstacleMap, obstacleViewedMap, currentMapState
    
    def resetTotalArea(self):
        obstacleMap = self.obstacleMap
        obstacleViewedMap = copy.deepcopy(obstacleMap)
        for agent in self.agents:
            obstacleViewedMap = vsb.updateVsbPolyOnImg([agent.getState()[0]],obstacleViewedMap)
        
        agentPos = [agent.getState()[0] for agent in self.agents]
        gPos = self.cartesian2Grid(agentPos)
        currentMapState = self.updatePosMap(gPos, obstacleViewedMap)
        return obstacleMap, obstacleViewedMap, currentMapState
    
    def initAgents(self, n):
        agents = []
        for i in range(0,n):
            agents.append(Agent())
        return agents
      
    def reset(self):
        self.agents = self.initAgents(len(self.agents))
        # need to update initial state for reset function
        self.obstacleMap,self.obsPlusViewed, self.currentMapState = self.initTotalArea()
        self.prevUnviewedCount = np.count_nonzero(self.currentMapState==0)
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
            curState = agent.getState()
            futureState = copy.deepcopy(curState[0])
            if action == 0:
                    pass
            elif action == 1:
                futureState[1] += 1
            elif action == 2:
                futureState[0] += -1
            elif action == 3:
                futureState[1] += -1
            elif action == 4:
                futureState[0] += 1
            # check if agent in obstacle
            isInObs = False
            for obs in vsb.obsPolyList:
                val = vsb.isPtinPoly(futureState,obs)
                if  val == 1 or val == 0:
                    isInObs = True
            if 0<futureState[0] <50 and 0<futureState[1] <50 and not isInObs:
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
            else:
                posOut.append(curState[0])
                velOut.append(curState[1])
        return posOut, velOut
    
    def step(self, agentActions):
        agentPos, agentVel = self.stepAgent(agentActions)
        gPos = self.cartesian2Grid(agentPos)
        # get new visibility and update obsPlusViewed
        self.obsPlusViewed = vsb.updateVsbPolyOnImg(agentPos,self.obsPlusViewed)
        # update position on currentMapState
        self.currentMapState = self.updatePosMap(gPos, self.obsPlusViewed)
        display = self.currentMapState
        # update reward mechanism
        reward = self.getReward()
        done = np.count_nonzero(self.currentMapState==0) == 0
        return agentPos, display, reward, done
                

    def render(self):
        img = copy.deepcopy(self.currentMapState)
        img = np.rot90(img,1)
        r = np.where(img==150, 255, 0)
        g = np.where(img==100, 255, 0)
        
        b = np.zeros_like(img)
        b_n = np.where(img==255, 100, 0)
        bgr = np.stack((b,g,r),axis = 2)
        bgr[:,:,0] = b_n
        displayImg = cv2.resize(bgr,(700,700),interpolation = cv2.INTER_AREA)
        
        cv2.imshow("Position Map", displayImg)
        cv2.imshow("raw", cv2.resize(img,(700,700),interpolation = cv2.INTER_AREA))
        pass
    
            
    def updatePosMap(self, gPos, obsPlusViewed):
        currMapState = copy.deepcopy(obsPlusViewed)
        for pos in gPos:
            currMapState[pos[0],pos[1]] = 100
        return currMapState
            
        
    def getReward(self):
        curUnviewedCount = np.count_nonzero(self.currentMapState==0)
        reward = self.prevUnviewedCount - curUnviewedCount
        self.prevUnviewedCount = curUnviewedCount
        return reward
        
    def cartesian2Grid(self, posList):
        gridList = []
        for pos in posList:
            _x = math.floor(pos[0]/CONST.GRID_SZ)
            _y = math.floor(pos[1]/CONST.GRID_SZ)
            gridList.append([int(_x),int(_y)])
        return gridList
        
                    


