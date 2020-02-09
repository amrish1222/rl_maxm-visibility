	# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adheesh
# author Bala Murali
# Copyright
# Constants

RENDER_PYGAME = True



## Drone Constants
class CONSTANTS:
    def __init__(self):
        self.TIME_STEP=1
        
        
        self.NUM_AGENTS = 1
        self.MAX_AGENT_VEL= 1   
        
        self.GRID_SZ = self.MAX_AGENT_VEL * self.TIME_STEP * 1.0

        self.MAX_STEPS = 50
        
        self.MAP_SIZE = self.MAX_STEPS * self.GRID_SZ

## Area
        


