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

        print('Initial Neural Net Call')
        
    def add_Layer(self,input_d,output_d):
        self.NNLayer.append(nl.Layer(input_d,output_d))

    def print_Layer(self):
        for i in range(len(self.NNLayer)):
            print('Layer ',i,end=" ")
            for j in range(len(self.NNLayer[i].NNCell)):
                print('Cell ',j)
                for k in range(len(self.NNLayer[i].NNCell[j].weights)):
                    print('w',k,' val: ',self.NNLayer[i].NNCell[j].weights[k])

    