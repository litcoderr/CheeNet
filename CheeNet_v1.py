'''
Version 1.0 CheeNet

Developed By James Chee(Youngchae Chee) @Litcoderr
You are welcome to contribute!!
'''
import numpy as np
import CheeLayer_v1 as nl # Custom Neural Layer Class

class NeuralNet:

    def __init__(self):
        # Layer Storage: Can access Every layer of this Net
        self.NNLayer = []

        # Training Data Sets

        self.training_input = []
        self.training_output = []

        print('Initial Neural Net Call')
        
    def add_Layer(self,layer_type,input_d,output_d):
        self.NNLayer.append(nl.Layer(layer_type,input_d,output_d))

    def add_trainingSet(self,training_input,training_output):
        self.training_input = training_input
        self.training_output = training_output

    def feedforward(self):
        #TODO make feedforwarding funciton
        print('I feed forward')

    def print_Layer(self):
        for i in range(len(self.NNLayer)):
            print('Layer ',i,end=" ")
            for j in range(len(self.NNLayer[i].NNCell)):
                print('Cell ',j)
                for k in range(len(self.NNLayer[i].NNCell[j].weights)):
                    print('w',k,' val: ',self.NNLayer[i].NNCell[j].weights[k])
