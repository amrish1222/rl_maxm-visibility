# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:14:44 2020

@author: amris
"""
import skgeom as sg
import numpy as np
from constants import CONSTANTS as K
from matplotlib.path import Path
CONST = K()


class Obstacle:
    def __init__(self):
        pass
    
    def getObstacleMap(self, emptyMap, vsb):
        obsList = self.getObstacles()
        for obs, isHole in obsList:
            vsb.addGeom2Arrangement(obs)

        # get obstacle polygon
        points = CONST.GRID_CENTER_PTS
        img = np.zeros_like(emptyMap, dtype = bool)
        for obs, isHole in obsList:
            p = Path(obs)
            grid = p.contains_points(points)
            mask = grid.reshape(50,50)
            img = np.logical_or(img , (mask if not isHole else np.logical_not(mask)))
           
        img = img.T
        img = np.where(img,150,emptyMap)
        return img, vsb
    
    def getObstacles(self):
        obstacle = self.obstacle2()
        return obstacle
    
    def obstacle1(self):
        obsList = []
        # add points in CW order and 
        isHole = False
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
        obsList.append([geom, isHole])
        
        geom = [[10,50],
                [10,45],
                [25,45],
                [25,46],
                [11,46],
                [11,50]]
        obsList.append([geom, isHole])

        
        geom = [[5,19],
                [5,35],
                [6,35],
                [6,20],
                [7,20],
                [7,19]]
        obsList.append([geom, isHole])
        
        return obsList
        
    
    def obstacle2(self):
        obsList = []
        # add points in CW order and 
        isHole = False
        geom = [[6,6],
                [6,12],
                [44,12],
                [44,6]]
        obsList.append([geom, isHole])
        
        geom = [[35,18],
                [35,23],
                [41,23],
                [41,18]]
        obsList.append([geom, isHole])
        
        geom = [[39,29],
                [39,34],
                [44,34],
                [44,29]]
        obsList.append([geom, isHole])
        
        geom = [[12,29],
                [12,39],
                [17,39],
                [17,29]]
        obsList.append([geom, isHole])
        
        geom = [[23,39],
                [23,44],
                [28,44],
                [28,39]]
        obsList.append([geom, isHole])
        
        return obsList
    
    
    def obstacle3(self):
        obsList = []
        # add points in CW order and 
        isHole = True
        geom = [[1,1],
                [1,49],
                [49,49],
                [49,1]]
        obsList.append([geom, isHole])
        
        isHole = False
        geom = [[6,6],
                [6,12],
                [44,12],
                [44,6]]
        obsList.append([geom, isHole])
        
        geom = [[35,18],
                [35,23],
                [41,23],
                [41,18]]
        obsList.append([geom, isHole])
        
        geom = [[39,29],
                [39,34],
                [44,34],
                [44,29]]
        obsList.append([geom, isHole])
        
        geom = [[12,29],
                [12,39],
                [17,39],
                [17,29]]
        obsList.append([geom, isHole])
        
        geom = [[23,39],
                [23,44],
                [28,44],
                [28,39]]
        obsList.append([geom, isHole])
        
        return obsList