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

from agent import Agent as Adversary

np.set_printoptions(precision=3, suppress=True)
class Env:
    def __init__(self):
        self.isNewSess = True
        self.timeStep = CONST.TIME_STEP
        self.obsMaps, self.vsbs, self.vsbPolys = self.initObsMaps_Vsbs()
        self.obstacleMap , self.vsb, self.vsbPoly, self.mapId = self.setRandMap_vsb()
        self.obstacleMap,self.obsPlusViewed, self.currentMapState, self.agents, self.adversaries = self.initTotalArea_agents(CONST.NUM_AGENTS)
        self.prevUnviewedCount = np.count_nonzero(self.currentMapState==0)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(f"checkpoints/cnn1.avi",self.fourcc, 50, (700,700))
    
    def initObsMaps_Vsbs(self):
        return obsMap.getAllObs_vsbs(np.zeros((50,50)))
    
    def setRandMap_vsb(self):
        i = random.randint(0, len(self.obsMaps)-1)
        return self.obsMaps[i], self.vsbs[i], self.vsbPolys[i], i
    
    def initTotalArea_agents(self, numAgents):
        # unviewed = 0
        # viewed = 255
        # obstacle = 150
        # agent Pos = 100
        # adversary Pos = 200
        
        obstacleMap = self.obstacleMap
        obstacleViewedMap = np.copy(obstacleMap)
        
        #initialize agents at random location
        agents = []
        x,y = np.nonzero(obstacleMap == 0)
        ndxs = random.sample(range(x.shape[0]), CONST.NUM_AGENTS + CONST.NUM_ADVRSRY)

        for ndx in ndxs[:-(CONST.NUM_ADVRSRY)]:
            agents.append(Agent(x[ndx]+0.5, y[ndx]+0.5))
#            agents.append(Agent())

        adversaries = []
        for ndx in ndxs[-(CONST.NUM_ADVRSRY):]:
            adversaries.append(Adversary(x[ndx]+0.5, y[ndx]+0.5))
#            adversaries.append(Adversary(25.5,10.5))
                   
        for agent in agents:
            obstacleViewedMap = self.vsb.updateVsbOnImg([agent.getState()[0]],obstacleViewedMap, self.vsbPoly)
        
        agentPos = [agent.getState()[0] for agent in agents]
        gPos = self.cartesian2Grid(agentPos)
        temp = self.updatePosMap(gPos, obstacleViewedMap, 100)
        
        advrsyPos = [adversary.getState()[0] for adversary in adversaries]
        advPos = self.cartesian2Grid(advrsyPos)
        currentMapState = self.updatePosMap(advPos, temp, 200)
        
        return obstacleMap, obstacleViewedMap, currentMapState, agents, adversaries
    
    def resetTotalArea(self):
        obstacleMap = self.obstacleMap
        obstacleViewedMap = np.copy(obstacleMap)
        for agent in self.agents:
            obstacleViewedMap = self.vsb.updateVsbOnImg([agent.getState()[0]],obstacleViewedMap, self.vsbPoly)
        
        agentPos = [agent.getState()[0] for agent in self.agents]
        gPos = self.cartesian2Grid(agentPos)
        temp = self.updatePosMap(gPos, obstacleViewedMap, 100)
        
        advrsyPos = [adversary.getState()[0] for adversary in self.adversaries]
        advPos = self.cartesian2Grid(advrsyPos)
        currentMapState = self.updatePosMap(advPos, temp, 200)
        
        return obstacleMap, obstacleViewedMap, currentMapState
    
    def initAgents(self, n):
        agents = []
        for i in range(0,n):
            agents.append(Agent())
        return agents
      
    def reset(self):
        
        # need to update initial state for reset function
        self.obstacleMap , self.vsb, self.vsbPoly, self.mapId = self.setRandMap_vsb()
        self.obstacleMap,self.obsPlusViewed, self.currentMapState, self.agents, self.adversaries = self.initTotalArea_agents(CONST.NUM_AGENTS)

        self.prevUnviewedCount = np.count_nonzero(self.currentMapState==0)
        
        advrsyPos = [adversary.getState()[0] for adversary in self.adversaries]
        
        state = []
        for agent in self.agents:
            state.append([agent.getState()[0], advrsyPos, self.currentMapState])
        
        return state
        
    def getActionSpace(self):
        return [0,1,2,3,4]
    
    def getStateSpace(self):
        return self.obstacleMap.size
    
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
            isValidPt = False
            if 0<futureState[0] <50 and 0<futureState[1] <50 :
                if self.obstacleMap[int(futureState[0]), int(futureState[1])] == 150:
                    isValidPt = True
            if 0<futureState[0] <50 and 0<futureState[1] <50 and not isValidPt:
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
        self.obsPlusViewed = self.vsb.updateVsbOnImg(agentPos,self.obsPlusViewed, self.vsbPoly)
        # update position on currentMapState
        temp = self.updatePosMap(gPos, self.obsPlusViewed, 100)
        
        advrsyPos = [adversary.getState()[0] for adversary in self.adversaries]
        advPos = self.cartesian2Grid(advrsyPos)
        self.currentMapState = self.updatePosMap(advPos, temp, 200)
        
#        AdvVisibility = self.vsb.checkPtInVsbPolyDict(advrsyPos, agentPos, self.vsbPoly)
        
        separation = np.linalg.norm(advrsyPos[0]-agentPos[0])
        AdvVisibility = separation <= CONST.SEPERATION_PENALTY
        
        display = self.currentMapState
        # update reward mechanism
        newAreaVis, penalty = self.getReward(AdvVisibility)
        reward = newAreaVis + penalty
        done = np.count_nonzero(self.currentMapState==0) == 0
        return agentPos, advrsyPos, display, reward, newAreaVis, penalty, done
                

    def render(self):
        img = np.copy(self.currentMapState)
        img = np.rot90(img,1)
        r = np.where(img==150, 255, 0)
        g = np.where(img==100, 255, 0)
        
        b = np.zeros_like(img)
        b_n = np.where(img==255, 100, 0)
        bgr = np.stack((b,g,r),axis = 2)
        bgr[:,:,0] = b_n
        
        # adversary Pos
        advPos = np.where(img == 200)
        bgr[advPos[0], advPos[1],1] = 255
        bgr[advPos[0], advPos[1],2] = 255
        
        displayImg = cv2.resize(bgr,(700,700),interpolation = cv2.INTER_AREA)
        
        cv2.imshow("Position Map", displayImg)
        cv2.waitKey(1)
    
    def save2Vid(self):
        img = np.copy(self.currentMapState)
        img = np.rot90(img,1)
        r = np.where(img==150, 255, 0)
        g = np.where(img==100, 255, 0)
        
        b = np.zeros_like(img)
        b_n = np.where(img==255, 100, 0)
        bgr = np.stack((b,g,r),axis = 2)
        bgr[:,:,0] = b_n
        
        # adversary Pos
        advPos = np.where(img == 200)
        bgr[advPos[0], advPos[1],1] = 255
        bgr[advPos[0], advPos[1],2] = 255
        
        displayImg = cv2.resize(bgr,(700,700),interpolation = cv2.INTER_AREA)
        
        self.out.write(displayImg.astype('uint8'))
#        cv2.imshow("raw", displayImg)
#        cv2.waitKey(1)
            
    def updatePosMap(self, gPos, obsPlusViewed, val):
        currMapState = np.copy(obsPlusViewed)
        for pos in gPos:
            currMapState[pos[0],pos[1]] = val
        return currMapState
        
        
    def getReward(self, AdvVisibility):
        curUnviewedCount = np.count_nonzero(self.currentMapState==0)
        newAreaVis = self.prevUnviewedCount - curUnviewedCount
        self.prevUnviewedCount = curUnviewedCount
        
        if newAreaVis < 0:
            print("Error calculating newArea")
        
        penalty = 0
        if AdvVisibility:
            penalty += -100
#            print("Visible")
        else:
#            print("Not Visible")
            pass
        return newAreaVis, penalty
        
    def cartesian2Grid(self, posList):
        gridList = []
        for pos in posList:
            _x = math.floor(pos[0]/CONST.GRID_SZ)
            _y = math.floor(pos[1]/CONST.GRID_SZ)
            gridList.append([int(_x),int(_y)])
        return gridList
        
                    


