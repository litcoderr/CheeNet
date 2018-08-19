'''
This Experiment shows simple Neural Network fit "y=x" equation

Developed By James Chee(Youngchae Chee) @Litcoderr
You are welcome to contribute!!
'''
import numpy as np
import CheeNet_v1 as CheeNet

if __name__ == '__main__':
    NN = CheeNet.NeuralNet()
    NN.add_Layer(2,3)
    NN.add_Layer(3,1)
    NN.add_Layer(1,1)
    NN.print_Layer()