"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

# standard libraries
import random

# third party libraries
import numpy as np


class Network(object):

    def __init__(self, sizes):
        """
        The list `sizes` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers.
        """

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) for (x, y) in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        """ Returns output by network if `a` is input of ndarray """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The `mini_batch` is a list of tuples `(x, y)`, and `eta`
        is the learning rate.
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        mini_batch_size = len(mini_batch)
        self.weights = [
            w - eta * nw / mini_batch_size
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - eta * nb / mini_batch_size
            for b, nb in zip(self.biases, nabla_b)
        ]


    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        """
        Train the neural network using mini-batch stochastic
        gradient descent.  The `training_data` is a list of tuples
        `(x, y)` representing the training inputs and the desired
        outputs. The variables `epochs` and `mini_batch_size` are
        what you'd expect - the number of epochs to train for, and the
        size of the mini-batches to use when sampling. `eta` is the
        learning rate, eta. If `test_data` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        """

        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test
                )
            else:
                print "Epoch {0} complete".format(j)


# Other functions
def sigmoid(z):
    """ The sigmoid function """
    return 1.0 / (1.0 + np.exp(-z))
