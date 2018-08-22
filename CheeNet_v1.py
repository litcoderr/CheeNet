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
                self.NNLayer[layerIndex].feedingProcess(self.NNLayer[layerIndex-1])
        else:
            print('Error: feeforward() --> Input_X has invalid dimension')

    def calculate_loss(self,input_x,target_y):
        # Check if input_x is valid
        self.valid = True
        if len(input_x) != self.NNLayer[0].input_dimension:
            self.valid = False
        #Check if target_y is valid
        if len(target_y) != self.NNLayer[-1].input_dimension:
            self.valid = False
        
        if self.valid:
            #for softmax loss
            if self.NNLayer[-1].layer_type == 'softmax':
                print('do softmax loss with logits')
        else:
            print('Error calculate_loss() --> Wrong input or output dimensions')

############# Debugging Methods ################
    def print_Result(self):
        for i in range(len(self.NNLayer[-1].NNCell)):
            print(self.NNLayer[-1].NNCell[i].In,end=" ")

    def print_Layer(self):
        for i in range(len(self.NNLayer)):
            for j in range(len(self.NNLayer[i].NNCell)):
                print((i,j),' In: ',self.NNLayer[i].NNCell[j].In,'Out: ',self.NNLayer[i].NNCell[j].Out)
                for k in range(len(self.NNLayer[i].NNCell[j].weights)):
                    print('w',k,' val: ',self.NNLayer[i].NNCell[j].weights[k])

    def print_training_data(self):
        print("--train Data--")
        print("X: ",self.training_input," ,Y: ",self.training_output)