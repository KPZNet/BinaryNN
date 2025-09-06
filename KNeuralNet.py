import copy
from datetime import *
import matplotlib.pyplot as plt
import numpy as np


# CHANGE

# akjf;alksdjf

#Print error rate over epochs to see training trend
def plot_error(epocs, error_history) :
    plt.plot ( epocs, error_history, label="errors" )
    plt.title ( "Training" )
    plt.xlabel ( "EPOCs" )
    plt.ylabel ( "error" )
    plt.legend ()
    plt.show ()

#Plot weight updates to see training trends over epochs
def plot_weights(epocs, weight_history) :
    wh = np.array ( weight_history )
    s = wh.shape[1]
    [plt.plot ( epocs, wh[:, i], label="weight {0}".format ( i ) ) for i in range ( s )]
    plt.title ( "Weights" )
    plt.xlabel ( "EPOCs" )
    plt.ylabel ( "weight" )
    plt.legend ()
    plt.show ()

#Sigmoid function
def sigmoidA(x) :
    return 1 / (1 + np.exp ( -x ))

#Sigmoid function derivative
def sigmoidA_derivative(x) :
    return x * (1 - x)

#Neural net class
#Uses replaceable activation functions
class NeuralNetwork :
    def __init__(self) :
        self.error_history = []
        self.epoch_list = []
        self.weight_history = []
        self.stop_delta = 0.001

    #Train network by backpropogating through epochs and adjust
    #weights each epoch
    def train(self, training_input, training_output, epochs=1000) :
        self.weights = np.array ( np.random.normal ( size=(col, 1) ) )
        for epoch in range ( epochs ) :
            stop = self.run_epoch ( training_input, training_output, epoch )
            if stop :
                break
    #Run one single epoch
    def run_epoch(self, training_input, training_output, epoch) :
        stop = False
        self.feed_forward ( training_input, sigmoid_fn=sigmoidA )
        self.backpropagation ( training_input, training_output, sigmoid_fn_derivative=sigmoidA_derivative )
        #Check error delta - if small enough, stop training
        err = np.average ( np.abs ( self.error ) )
        if err < self.stop_delta :
            stop = True
        #Add errors and weights to history list for later plotting
        self.error_history.append ( err )
        self.epoch_list.append ( epoch )
        return stop
    #Run a feedfoward cycle to calculate hidden vector
    def feed_forward(self, training_input, sigmoid_fn) :
        self.hidden = sigmoid_fn ( np.dot ( training_input, self.weights ) )
    #Run a back propogation to train net
    def backpropagation(self, training_input, training_output, sigmoid_fn_derivative) :
        self.error = training_output - self.hidden
        delta = self.error * sigmoid_fn_derivative ( self.hidden )
        self.weights += np.dot ( training_input.T, delta )
        self.weight_history.append ( copy.deepcopy ( self.weights ) )
    #Predict a result from trained network
    def predict(self, sigmoid_fn, new_input) :
        prediction = sigmoid_fn ( np.dot ( new_input, self.weights ) )
        return prediction
#Run a test set of random vectors through network to see how
#well we perform
def test_run_random(nnet, row, col):
    inputs = np.random.randint ( 2, size=(row, col) )
    print("\nTEST RESULTS")
    for input in inputs:
        e = input[2]
        print ( 'Input: ', input, 'Expected:', e, ' , NN Result: ', nnet.predict ( sigmoid_fn=sigmoidA, new_input=input ) )

#Seed random number generator to something new
np.random.seed(datetime.now().microsecond)
#Initialize the input binary arrays to rows and columns
row, col = 100, 8

#Tweak the output so it matches the input's 3rd column
#We want a certain input column to match output vector to where
#we have a known input/output vector set to train network
inputsA = np.random.randint ( 2, size=(row, col) )
outputsA = np.array ( [inputsA[:, 2]] ).T

#Create neural net
NNN = NeuralNetwork ()
#Train network using input/output known set vectors
NNN.train ( inputsA, outputsA )

#Plot the error rates through epochs
plot_error ( NNN.epoch_list, NNN.error_history )
#Plot weights through epochs
plot_weights ( NNN.epoch_list, NNN.weight_history )

#Run a random test set through network to see how well it does
test_run_random(NNN, 5, col)
