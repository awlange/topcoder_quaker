"""network3.py
~~~~~~~~~~~~~~

A Theano-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.

Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.

This program is loosely based on the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), as well as some ideas
of Chris Olah (http://colah.github.io ).

"""

#### Libraries
# Standard library
# import cPickle
import gzip

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor.signal import downsample

import random

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

import json
import pickle


#### Constants
GPU = False
if GPU:
    print("Trying to run under a GPU.  If this is not desired, then modify "+\
        "network3.py\nto set the GPU flag to False.")
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'

# #### Load the MNIST data
# def load_data_shared(filename="../data/mnist.pkl.gz"):
#     f = gzip.open(filename, 'rb')
#     training_data, validation_data, test_data = cPickle.load(f)
#     f.close()
#     def shared(data):
#         """Place the data into shared variables.  This allows Theano to copy
#         the data to the GPU, if one is available.
#
#         """
#         shared_x = theano.shared(
#             np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
#         shared_y = theano.shared(
#             np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
#         return shared_x, T.cast(shared_y, "int32")
#     return [shared(training_data), shared(validation_data), shared(test_data)]

#### Main class used to construct and train networks
class Network():

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output, self.mini_batch_size)
        self.output = self.layers[-1].output

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)//mini_batch_size
        num_validation_batches = size(validation_data)//mini_batch_size
        num_test_batches = size(test_data)//mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.log_likelihood()+0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        best_iteration = 0
        test_accuracy = 0

        # Do the actual training
        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            if epoch % 10 == 0:
                save_network(self, "output/checkpoint_" + str(epoch) + ".json")
            for minibatch_index in range(int(num_training_batches)):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 100 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        test_accuracy = np.mean(
                            [test_mb_accuracy(j) for j in range(num_test_batches)])
                        print('The corresponding test accuracy is {0:.2%}'.format(
                            test_accuracy))
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
              best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

    def log_likelihood(self):
        """Return the log-likelihood cost."""
        return -T.mean(T.log(self.output)[T.arange(self.y.shape[0]), self.y])


#### Define layer types

class ConvPoolLayer():
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.inpt = None
        self.output = None
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        pooled_out = downsample.max_pool_2d(
            input=conv_out, ds=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))


class FullyConnectedLayer():

    def __init__(self, n_in, n_out, activation_fn=sigmoid):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.inpt = None
        self.output = None
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer():

    def __init__(self, n_in, n_out):
        self.inpt = None
        self.output = None
        self.n_in = n_in
        self.n_out = n_out
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax(T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]


def vectorized_result3(j):
    e = np.zeros((1393, 1))  # zeroth element corresponds to not happening in foreseeable future, others hour buckets
    # print("Setting: {}".format(j))
    e[j] = 1.0
    return e


def divide_raw_data(raw_data):
    positive_data = []
    negative_data = []
    for x, y in raw_data:
        if y > 0:
            # positive_data.append((x, vectorized_result3(y)))
            positive_data.append((x, y))
        else:
            # negative_data.append((x, vectorized_result3(0)))
            negative_data.append((x, 0))
    return positive_data, negative_data


def split_data(data, x=0.8):
    # 80-20 split of data, randomly shuffled. but x can be different if you want.
    random.shuffle(data)
    n80 = int(len(data) * x)
    return data[:n80], data[n80:]


def shared(data_x, data_y):
    """Place the data into shared variables.  This allows Theano to copy
    the data to the GPU, if one is available.
    """
    shared_x = theano.shared(
        np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(
        np.asarray(data_y, dtype=theano.config.floatX), borrow=True)
    return shared_x, T.cast(shared_y, "int32")


def main():

    n_epochs = 20
    network_sizes = [392, 45, 1393]
    mini_batch_size = 20

    # Network
    network = load2("output/bootstrap_network_2.pickle", mini_batch_size)
    # network = Network([FullyConnectedLayer(392, 45),
    #                    SoftmaxLayer(45, 1393)], 20)

    restart = True

    # Read in out data for fitting
    raw_data = []

    with open("../train_1/transformed/combined/combined_pos_hi_transformed.csv") as f:
        n_line = 0
        for line in f:
            n_line += 1
            if n_line > 1:

                split_line = [float(s) for s in line.split(",")]
                hour_diff = int(split_line[1])
                if hour_diff < 1393:  # Skip early ones where diff is too high
                    raw_data.append((np.reshape(split_line[2:], (len(split_line[2:]), 1)), hour_diff))


    print("Read data point: {}".format(len(raw_data)))

    # Seperate data into positive and negative data, test and train
    positive_data, negative_data = divide_raw_data(raw_data)

    print("Positive data points: {}".format(len(positive_data)))
    print("Negative data points: {}".format(len(negative_data)))

    # 5-fold bootstrapping
    for bootstrap in [2]:
        # try:
            print("Bootstrap: {}".format(str(bootstrap)))

            # train_positive_data, test_positive_data = split_data(positive_data, 1.0)
            train_positive_data = positive_data
            test_positive_data = positive_data
            train_negative_data, test_negative_data = split_data(negative_data, 0.0005)  # Don't need all those negatives
            # test_data = test_positive_data + test_negative_data
            # # Convert test data to work with network
            # tmp = []
            # for x, y in test_data:
            #     yval = 1 if y[0] > 0.0 else 0
            #     tmp.append((x, yval))
            # test_data = tmp

            # For training, we bootstrap the positive data so that the negative data doesn't dwarf the positive
            # Keep at least one copy of the train positive data
            # bootstrapped_positive_data = bootstrap_data(train_positive_data, len(train_negative_data) // 2) + train_positive_data
            bootstrapped_positive_data = train_positive_data
            train_data = bootstrapped_positive_data + train_negative_data

            print("Training set: {} {}".format(len(bootstrapped_positive_data), len(train_negative_data)))

            tx = []
            ty = []
            for x, y in train_data:
                tx.append(x)
                ty.append(y)
            tx = np.reshape(tx, (len(tx), 392))
            # tx = np.asarray(tx)
            ty = np.reshape(ty, (len(ty),))
            # ty = np.asarray(ty)
            train_data = shared(tx, ty)

            # Start the fit
            network.SGD(train_data, n_epochs, mini_batch_size, 1.0, train_data, train_data, lmbda=0.001)

            print("Epochs done. Writing data to disk...")

            # Save the network for later
            save_network2(network, "output/bootstrap_network_" + str(bootstrap) + ".pickle")
            save_network(network, "output/bootstrap_network_" + str(bootstrap) + ".json")

        # except Exception as e:
        #     print("wtf, mate: {}", str(e))


def save_network(network, filename):
    data = {
        "sizes": [layer.n_in for layer in network.layers] + [network.layers[-1].n_out],
        "biases": [[[b] for b in network.layers[i].b.container.data.tolist()] for i in range(len(network.layers))],
        "weights": [[w for w in network.layers[i].w.container.data.transpose().tolist()] for i in range(len(network.layers))]
    }
    f = open(filename, "w")
    json.dump(data, f)
    f.close()

def save_network2(network, filename):
    f = open(filename, "wb")
    pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)
    f.close()


def load2(filename, mini_batch_size):
    f = open(filename, "rb")
    net = pickle.load(f)
    f.close()
    return net

def load(filename, mini_batch_size):
    f = open(filename, "r")
    data = json.load(f)
    f.close()

    sizes = data["sizes"]
    layers = []
    for i in range(len(sizes)-2):
        layers.append(FullyConnectedLayer(sizes[i], sizes[i+1]))
    layers.append(FullyConnectedLayer(sizes[-2], sizes[-1]))
    # layers.append(SoftmaxLayer(sizes[-2], sizes[-1]))
    net = Network(layers, mini_batch_size)

    # Set weights and biases
    wts = [np.array(w) for w in data["weights"]]
    biases = [np.array(b) for b in data["biases"]]

    for i, layer in enumerate(net.layers):
        net.layers[i].w = theano.shared(
            np.asarray(
                np.reshape(wts[i], (layer.n_in, layer.n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)

        net.layers[i].b = theano.shared(
            np.asarray(np.reshape(biases[i], (layer.n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)

    return net


def test2():
    network = load("foobs.json", 10)
    save_network(network, "doobs.json")


def test():

    network = Network([FullyConnectedLayer(2, 2),
                       SoftmaxLayer(2, 1)], 1)

    save_network(network, "foobs.json")


def read_pickle_write_json():
    pnet = load2("best.pickle", 100)
    save_network(pnet, "best.json")


if __name__ == "__main__":
    # main()
    # test()
    # test2()
    read_pickle_write_json()
