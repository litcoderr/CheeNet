'''
Version 1.0 CheeNet

Developed By James Chee(Youngchae Chee) @Litcoderr
You are welcome to contribute!!
'''
import CheeLayer_v1 as nl # Custom Neural Layer Class

class NeuralNet:

    def __init__(self):
        # Layer Storage: Can access Every layer of this Net
        self.NNLayer = []

        # Training Data Sets
        self.training_input = []
        self.training_output = []

        # Error Detection variables
        self.addingLayer_ErrorSwitch = True
        self.currentNetdim = 0

        print('Initial Neural Net Call Made')
        
    def add_Layer(self,layer_type,input_d,output_d):
        # 1. Stop when Error occurse during previous layer addition
        if self.addingLayer_ErrorSwitch:
            # 2. Check if it is OK to add Layers
            self.canAddLayer = False
            if len(self.NNLayer) == 0:
                self.canAddLayer = True
                self.currentNetdim = output_d
            else:
                if input_d == self.currentNetdim:
                    self.canAddLayer = True
                    self.currentNetdim = output_d
            
            # 3. If OK->AddLayer NotOK->StopAdding and Display Error code
            if self.canAddLayer:
                self.NNLayer.append(nl.Layer(layer_type,input_d,output_d))
            else:
                self.addingLayer_ErrorSwitch = False
                print('Error: add_Layer() --> input_d should be ',self.currentNetdim)
        
        
    def add_trainingSet(self,training_input,training_output):
        self.training_input = training_input
        self.training_output = training_output

    def feedforward(self,input_x):
        # Check if input_x is valid
        self.valid = False
        if len(input_x) == self.NNLayer[0].input_dimension:
            self.valid = True
        
        # If valid to feedforward
        if self.valid:
            # 1. Insert input_x to first layer
            for i in range(len(self.NNLayer[0].NNCell)):
                self.NNLayer[0].NNCell[i].In = input_x[i]
            # 2. Iterate to the last layer
            for layerIndex in range(1,len(self.NNLayer)):
                for cellIndex in range(len(self.NNLayer[layerIndex].NNCell)):
                    # temp to be stored
                    self.temp = 0
                    # previous layer cell number
                    self.prev_cell_num = self.NNLayer[layerIndex-1].input_dimension

                    # 1. calculate and store to temp
                    for i in range(self.prev_cell_num):
                        self.temp = self.temp+(self.NNLayer[layerIndex-1].NNCell[i].In*self.NNLayer[layerIndex-1].NNCell[i].weights[cellIndex])
                    
                    # 2. update In for this cell
                    self.NNLayer[layerIndex].NNCell[cellIndex].In = self.temp

        else:
            print('Error: feeforward() --> Input_X has invalid dimension')

############# Debugging Methods ################
    def print_Layer(self):
        for i in range(len(self.NNLayer)):
            for j in range(len(self.NNLayer[i].NNCell)):
                print((i,j),' In: ',self.NNLayer[i].NNCell[j].In,'Out: ',self.NNLayer[i].NNCell[j].Out)
                for k in range(len(self.NNLayer[i].NNCell[j].weights)):
                    print('w',k,' val: ',self.NNLayer[i].NNCell[j].weights[k])
