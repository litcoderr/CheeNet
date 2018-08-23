'''
This Experiment shows simple Neural Network fit "y=x" equation

Developed By James Chee(Youngchae Chee) @Litcoderr
You are welcome to contribute!!
'''
import CheeNet_v1 as CheeNet

if __name__ == '__main__':
    NN = CheeNet.NeuralNet()
    NN.add_trainingSet([[1,2],[2,3]],[[1,0],[0,1]])
    NN.add_Layer('sigmoid',2,3)
    NN.add_Layer('sigmoid',3,3)
    NN.add_Layer('sigmoid',3,2)
    NN.add_Layer('softmax',2,1)
    NN.feedforward([1,2])
    print(NN.calculate_loss([1,0]))
    NN.back_propagate([1,0])
    NN.print_derivative()