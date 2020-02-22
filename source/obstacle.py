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
        obstacle = self.obstacle1()
        return obstacle
    
    def obstacle1(self):
        obsList = []
        # add points in CW order and 
        
        geom = [[7,0],
                [7,20],
                [29,20],
                [29,29],
                [10,29],
                [10,40],
                [30,40],
                [30,39],
                [11,39],
                [11,30],
                [30,30],
                [30,19],
                [8,19],
                [8,0]]
        obsList.append(geom)
        
        geom = [[10,50],
                [10,45],
                [25,45],
                [25,46],
                [11,46],
                [11,50]]
        obsList.append(geom)
        
        geom = [[5,19],
                [5,35],
                [6,35],
                [6,20],
                [7,20],
                [7,19]]
        obsList.append(geom)
        
        return obsList
        
    
    def obstacle2(self):
        obsList = []
        # add points in CW order and 
        
        geom = [[6,6],
                [6,12],
                [44,12],
                [44,6]]
        obsList.append(geom)
        
        geom = [[35,18],
                [35,23],
                [41,23],
                [41,18]]
        obsList.append(geom)
        
        geom = [[39,29],
                [39,34],
                [44,34],
                [44,29]]
        obsList.append(geom)
        
        geom = [[12,29],
                [12,39],
                [17,39],
                [17,29]]
        obsList.append(geom)
        
        geom = [[23,39],
                [23,44],
                [28,44],
                [28,39]]
        obsList.append(geom)
        
        return obsList
    
    def obstacle3(self):
        obsList = []
        # add points in CW order and 
        
        geom = [[6,6],
                [6,12],
                [44,12],
                [44,6]]
        obsList.append(geom)
        
        geom = [[35,18],
                [35,23],
                [41,23],
                [41,18]]
        obsList.append(geom)
        
        geom = [[39,29],
                [39,34],
                [44,34],
                [44,29]]
        obsList.append(geom)
        
        geom = [[12,29],
                [12,39],
                [17,39],
                [17,29]]
        obsList.append(geom)
        
        geom = [[23,39],
                [23,44],
                [28,44],
                [28,39]]
        obsList.append(geom)
        
        return obsList