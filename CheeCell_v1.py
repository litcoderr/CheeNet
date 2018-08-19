'''
Version 1.0 CheeCell

Developed By James Chee(Youngchae Chee) @Litcoderr
You are welcome to contribute!!
'''

import numpy as np
import random

class Cell:
    
    def __init__(self,weightNum):
        self.weights = []
        self.weights_Number = weightNum
        self.forwarded_val = 0
        self.backwarded_val= 0

        # 1. initialize Weights
        self.initializeWeights()

    def initializeWeights(self):
        self.weights = []
        for _ in range(self.weights_Number):
            self.weights.append(1000*random.random())
