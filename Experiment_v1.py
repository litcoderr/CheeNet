'''
This Experiment shows simple Neural Network fit "y=x" equation

Developed By James Chee(Youngchae Chee) @Litcoderr
You are welcome to contribute!!
'''
import CheeNet_v1 as CheeNet

if __name__ == '__main__':
    # Configure Neural Net
    NN = CheeNet.NeuralNet(learning_rate=0.1,epochs=1000,limit=0.1)
    NN.add_trainingSet([[0,0],[1,0],[0,1],[1,1]],[[1,0],[0,1],[0,1],[1,0]])

    # Make the Architecture of the Network
    NN.add_Layer('sigmoid',2,2)
    NN.add_Layer('sigmoid',2,1)
    
    # Train the Network
    NN.train()

    # Test
    NN.feedforward([0,0])
    NN.print_Result()