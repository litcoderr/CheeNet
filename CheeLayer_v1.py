'''
Version 1.0 CheeLayer

Developed By James Chee(Youngchae Chee) @Litcoderr
You are welcome to contribute!!
'''
import CheeCell_v1 as nc #Custom Neural Cell Class
import math

class Layer:

    def __init__(self,layer_type,input_d,output_d):
        # Layer Type
        self.layer_type = layer_type

        # Dimensionality of this layer
        self.input_dimension = input_d
        self.output_dimension = output_d

        # Cell Storage: Can access Every Cell of this Layer
        self.NNCell = []

        # 1. Make the initial cells
        self.initializeCells()

    # Called when feeding is needed
    def feedingProcess(self,previous_layer):
        self.pre_layer = previous_layer
        for cellIndex in range(len(self.NNCell)):
            # temp to be stored
            self.temp = 0
            # previous layer cell number
            self.prev_cell_num = self.pre_layer.input_dimension

            # 1. calculate and store to temp
            for i in range(self.prev_cell_num):
                self.temp = self.temp+(self.pre_layer.NNCell[i].In*self.pre_layer.NNCell[i].weights[cellIndex])
            
            # 2. update In for this cell
            self.NNCell[cellIndex].In = self.temp
        
        # Feed based on the type of Layer
        if self.layer_type == 'sigmoid':
            for cellIndex in range(len(self.NNCell)):
                self.NNCell[cellIndex].In = 1/(1+math.exp(-self.NNCell[cellIndex].In))
        elif self.layer_type == 'softmax':
            self.temporary_sum = 0 # sum of every cell in the layer
            # 1. calculate sum
            for cellIndex in range(len(self.NNCell)):
                self.temporary_sum = self.temporary_sum + self.NNCell[cellIndex].In
            # 2. Update every cell with logits
            for cellIndex in range(len(self.NNCell)):
                self.NNCell[cellIndex].In = self.NNCell[cellIndex].In / self.temporary_sum
    
    # Called when loss is needed
    def lossProcess(self,target_y):
        self.temp_loss = 0
        # Softmax loss function
        if self.layer_type == 'softmax':
            for cell_index in range(len(self.NNCell)):
                self.temp_loss = self.temp_loss + (-1)*target_y[cell_index]*math.log(self.NNCell[cell_index].In)
        
        return self.temp_loss
    
    # Initialize all cells in this layer
    def initializeCells(self):
        for _ in range(self.input_dimension):
            self.NNCell.append(nc.Cell(self.output_dimension))