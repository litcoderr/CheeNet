'''
Version 1.0 CheeLayer

Developed By James Chee(Youngchae Chee) @Litcoderr
You are welcome to contribute!!
'''

import numpy as np
import CheeCell_v1 as nc #Custom Neural Cell Class

class Layer:

    def __init__(self,input_d,output_d):
        # Dimensionality of this layer
        self.input_dimension = input_d
        self.output_dimension = output_d

        # Cell Storage: Can access Every Cell of this Layer
        self.NNCell = []

        # 1. Make the initial cells
        self.initializeCells()

    def initializeCells(self):
        for _ in range(self.input_dimension):
            self.NNCell.append(nc.Cell(self.output_dimension))