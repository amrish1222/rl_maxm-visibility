# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:14:44 2020

@author: amris
"""
import skgeom as sg
from constants import CONSTANTS as K
CONST = K()

class Obstacle:
    def __init__(self):
        pass
    
    def getObstacleMap(self, emptyMap, vsb):
        obsList = self.getObstacles()
        for obs in obsList:
            vsb.addGeom2Arrangement(obs)
        # get obstacle polygon
        
        for i in range(0, int(CONST.MAP_SIZE)):
            for j in range(0, int(CONST.MAP_SIZE)):
                x = CONST.GRID_SZ/2 + i* CONST.GRID_SZ
                y = CONST.GRID_SZ/2 + j* CONST.GRID_SZ
                for obs in vsb.obsPolyList:
                    if vsb.isPtinPoly([x,y],obs) == 1:
                        emptyMap[i,j] = 150
        return emptyMap, vsb
    
    def getObstacles(self):
        obsList = []
        # add points in CW order and 
        
        geom = [[10,10],
                [10,15],
                [15,15],
                [15,10]]
        obsList.append(geom)
        
        geom = [[20,20],
                [20,30],
                [25,30],
                [25,25],
                [30,25],
                [30,20]]
        obsList.append(geom)
        
        return obsList
        