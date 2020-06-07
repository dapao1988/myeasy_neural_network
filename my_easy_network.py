#!/usr/bin/env python
# coding: utf-8


import numpy
# scipy.special for the sigmoid function expit()
import scipy.special


class MyEasyNetwork:


    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.__inodes = inputnodes
        self.__hnodes = hiddennodes
        self.__onodes = outputnodes

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        self.wih = numpy.random.normal(0.0, pow(self.__inodes, -0.5), (self.__hnodes, self.__inodes))
        self.who = numpy.random.normal(0.0, pow(self.__hnodes, -0.5), (self.__onodes, self.__hnodes))

        # learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass


    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot(
                (output_errors * final_outputs * (1.0 - final_outputs)),
                numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot(
                (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                numpy.transpose(inputs))

        pass


    # inference the neural network
    def inference(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


if __name__ == '__main__':
    # number of input, hidden and output nodes
    INPUT_NODES_NUM = 3
    HIDDEN_NODES_NUM = 3
    OUTPUT_NODES_NUM = 3
    # initialise a learning rate, better be a smaller value (not zero)
    LEARNING_RATE = 0.3

    # create instance of neural network
    neural_network = MyEasyNetwork(INPUT_NODES_NUM, HIDDEN_NODES_NUM, OUTPUT_NODES_NUM, LEARNING_RATE)

    # test inference (doesn't mean anything useful yet)
    print(neural_network.inference([1.0, 0.5, -1.5]))
