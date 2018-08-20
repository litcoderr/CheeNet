'''
This Experiment shows simple Neural Network fit "y=x" equation

Developed By James Chee(Youngchae Chee) @Litcoderr
You are welcome to contribute!!
'''
import CheeNet_v1 as CheeNet

if __name__ == '__main__':
    NN = CheeNet.NeuralNet()
    NN.add_Layer('sigmoid',2,3)
    NN.add_Layer('sigmoid',3,1)
    NN.feedforward([1,2])
    NN.print_Layer()
    NN.print_Result()