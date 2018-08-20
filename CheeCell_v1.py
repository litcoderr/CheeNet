'''
Version 1.0 CheeCell

Developed By James Chee(Youngchae Chee) @Litcoderr
You are welcome to contribute!!
'''
import random

class Cell:
    
    def __init__(self,weightNum):
        self.weights = []
        self.weights_Number = weightNum
        self.In = 0
        self.Out= 0

        # 1. initialize Weights
        self.initializeWeights()

    def initializeWeights(self):
        self.weights = []
        for _ in range(self.weights_Number):
            self.weights.append(10*random.random())