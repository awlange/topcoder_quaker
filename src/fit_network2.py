"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import json
import sys
import math

# Third-party libraries
import numpy as np

# Hard coded weighting. Positive points are 1000x more weighted
WT = {
    0: np.asarray([0.1]),
    1: np.asarray([10.0])
}

MAX_REGRESSION_VALUE = 4000.0

class QuadraticCost:

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime_vec(z)


class WeightedQuadraticCost:

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        wt = WT.get(1) if y[0] > 0.0 else WT.get(0)
        return 0.5*np.linalg.norm(wt * (a-y))**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        wt = WT.get(1) if y[0] > 0.0 else WT.get(0)
        return wt * (a-y) * sigmoid_prime_vec(z)


class CrossEntropyCost:

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.nan_to_num(np.sum(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return (a-y)


class WeightedCrossEntropyCost:

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        wt = WT.get(1) if y[0] > 0.0 else WT.get(0)
        return np.nan_to_num(np.sum(-wt * y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        wt = WT.get(1) if y[0] > 0.0 else WT.get(0)
        return wt * (a-y)


class Network:

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid_vec(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            bootstrap_index=0):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        n_data = len(evaluation_data) if evaluation_data else 0
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):

            # Write check point to disk every so often
            # Save the network for later
            if (j+1) % 10 == 0:
                print("Writing checkpoint...")
                self.save("output/checkpoint_" + str(bootstrap_index) + ".json")

            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {} = {}".format(accuracy, n, accuracy / n))
            if monitor_evaluation_cost:
                if j % 2 == 0:
                    cost = self.total_cost(evaluation_data, lmbda, convert=False)
                    # cost = self.total_cost(evaluation_data, lmbda, convert=True)
                    evaluation_cost.append(cost)
                    print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                # accuracy = self.accuracy(evaluation_data)
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))
            print()
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid_vec(z)
            activations.append(activation)
        # backward pass
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            spv = sigmoid_prime_vec(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def evaluate2(self, test_data):
        test_results = [(float(self.feedforward(x)[0]), float(y[0])) for (x, y) in test_data]
        return test_results

    def evaluate3(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return test_results

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
            # results = [(np.round(self.feedforward(x)), y) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
            # results = [(np.round(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    # cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=CrossEntropyCost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)


def vectorized_result(j):
    # e = np.zeros((2, 1))
    # e[j] = 1.0

    e = np.zeros((1, 1))
    if j > 0:
        e[0] = 1.0
    return e

def vectorized_result2(prob, hour):
    e = np.zeros((2, 1))
    e[0] = prob
    e[1] = hour
    return e

def vectorized_result3(j):
    e = np.zeros((1393, 1))  # zeroth element corresponds to not happening in foreseeable future, others hour buckets
    # print("Setting: {}".format(j))
    e[j] = 1.0
    return e

def vectorized_result4(hour):
    e = np.zeros((1, 1))
    e[0] = hour / MAX_REGRESSION_VALUE
    return e


def divide_raw_data(raw_data):
    positive_data = []
    negative_data = []
    for x, y in raw_data:
        if y > 0:
            positive_data.append((x, vectorized_result3(y)))
        else:
            negative_data.append((x, vectorized_result3(0)))
    return positive_data, negative_data


def split_data(data, x=0.8):
    # 80-20 split of data, randomly shuffled. but x can be different if you want.
    random.shuffle(data)
    n80 = int(len(data) * x)
    return data[:n80], data[n80:]


def bootstrap_data(data, n):
    """
    Resample (with replacement) the data for n times
    """
    return [random.choice(data) for _ in range(n)]


def main():

    n_epochs = 50
    network_sizes = [392, 45, 1393]

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

                    # raw_data.append((np.reshape(split_line[2:], (len(split_line[2:]), 1)),
                    #                  vectorized_result3(int(split_line[1]))))

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
            test_data = test_positive_data + test_negative_data
            # Convert test data to work with network
            tmp = []
            for x, y in test_data:
                yval = 1 if y[0] > 0.0 else 0
                tmp.append((x, yval))
            test_data = tmp

            # For training, we bootstrap the positive data so that the negative data doesn't dwarf the positive
            # Keep at least one copy of the train positive data
            # bootstrapped_positive_data = bootstrap_data(train_positive_data, len(train_negative_data) // 2) + train_positive_data
            bootstrapped_positive_data = train_positive_data
            train_data = bootstrapped_positive_data + train_negative_data

            print("Training set: {} {}".format(len(bootstrapped_positive_data), len(train_negative_data)))

            # Network
            network = Network(network_sizes)

            # If want to, can read in network data here
            if restart:
                network = load("output/bootstrap_network_" + str(bootstrap) + ".json")

            # Start the fit
            evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = network.SGD(
                train_data, n_epochs, 100, 1.0, 0.001,
                monitor_training_cost=True,
                # monitor_training_accuracy=True,
                # monitor_evaluation_cost=True,
                # monitor_evaluation_accuracy=True,
                # evaluation_data=test_data,
                bootstrap_index=bootstrap)

            print("Epochs done. Writing data to disk...")

            with open("output/train_cost_" + str(bootstrap) + ".dat", "w") as fc:
                for cost in training_cost:
                    fc.write(str(cost) + "\n")

            with open("output/accuracy_" + str(bootstrap) + ".dat", "w") as fa:
                accuracy_train = network.accuracy(train_data, convert=True)
                # accuracy_test = network.accuracy(test_data, convert=True)
                fa.write("Accuracy on train data: {} / {} = {}\n".format(accuracy_train, len(train_data), accuracy_train / len(train_data)))
                # fa.write("Accuracy on test data: {} / {} = {}\n".format(accuracy_test, len(test_data), accuracy_test / len(test_data)))

            # Save the network for later
            network.save("output/bootstrap_network_" + str(bootstrap) + ".json")

            if False:
                # Save data sets in case we want to continue training
                with open("output/train_data_" + str(bootstrap) + ".csv", "w") as ft:
                    ft.write("," + ",".join([str(s) for s in range(network_sizes[0])]))
                    ft.write("\n")
                    i = 0
                    for x, y in train_data:
                        s = str(i) + "," + str(float(y[0])) + ","
                        s += ",".join([str(float(val)) for val in x])
                        s += "\n"
                        ft.write(s)
                        i += 1

                with open("output/test_data_" + str(bootstrap) + ".csv", "w") as ft:
                    ft.write("," + ",".join([str(s) for s in range(network_sizes[0])]))
                    ft.write("\n")
                    i = 0
                    for x, y in test_data:
                        s = str(i) + "," + str(float(y)) + ","
                        s += ",".join([str(float(val)) for val in x])
                        s += "\n"
                        ft.write(s)
                        i += 1

        # except Exception as e:
        #     print("wtf, mate: {}", str(e))


def run():
    # Read in a network and run some sample data through it for testing

    networks = []
    networks.append(load("output/bootstrap_network_0.json"))
    networks.append(load("output/bootstrap_network_1.json"))
    networks.append(load("output/bootstrap_network_2.json"))
    networks.append(load("output/bootstrap_network_3.json"))
    networks.append(load("output/bootstrap_network_4.json"))

    raw_data = []
    with open("../train_1/test/data_transformed_6.csv") as f:
        n_line = 0
        for line in f:
            n_line += 1
            if n_line > 1:
                split_line = [float(s) for s in line.split(",")]
                raw_data.append((np.reshape(split_line[2:], (len(split_line[2:]), 1)), split_line[1]))

    print("Read data point: {}".format(len(raw_data)))

    # Seperate data into positive and negative data, test and train
    positive_data, negative_data = divide_raw_data(raw_data)
    all_data = positive_data + negative_data

    # Run network(s)
    results = []
    for i in range(len(networks)):
        # results.append(networks[i].evaluate2(positive_data))
        # results.append(networks[i].evaluate2(negative_data))
        results.append(networks[i].evaluate2(all_data))

    # Aggregate the predictions
    err = []
    aggs = []
    classify_err = 0
    wrongs = []
    for p0, p1, p2, p3, p4 in zip(results[0], results[1], results[2], results[3], results[4]):
        mean = (p0[0] + p1[0] + p2[0] + p3[0] + p4[0]) / 5.0
        max_val = np.max([p0[0], p1[0], p2[0], p3[0], p4[0]])
        min_val = np.min([p0[0], p1[0], p2[0], p3[0], p4[0]])

        # If more than 1 have high prob, use max
        # n_high_prob = 0
        # thresh = 0.98
        # if p0[0] > thresh:
        #     n_high_prob += 1
        # if p1[0] > thresh:
        #     n_high_prob += 1
        # if p2[0] > thresh:
        #     n_high_prob += 1
        #
        # agg = max_val if n_high_prob >= 2 else mean
        agg = mean

        aggs.append((p0[1], agg, p0[0], p1[0], p2[0], p3[0], p4[0]))
        err.append(np.abs(p0[1] - agg))

        ce = int(round(abs(round(agg) - p0[1])))
        classify_err += ce
        if ce > 0:
            wrongs.append((p0[1], agg, p0[0], p1[0], p2[0], p3[0], p4[0]))

    # for x in aggs:
    for x in wrongs:
        print(x)

    print("Mean absolute error: {}".format(sum(err) / len(err)))
    print("Classification error: {}".format(classify_err))


def test_network():

    sizes = [2, 2, 1]
    # network = Network(sizes)
    # network.save("test_network.json")
    network = load("test_network.json")

    act = [3.0, 4.0]
    result = network.feedforward(np.reshape(act, (2, 1)))
    print(result)

    # Dumb implementation of feedforward for ease of translation to C++
    a = act
    for i_layer in range(len(sizes)-1):
        b = []
        layer_weights = list(network.weights[i_layer])
        layer_biases = list(network.biases[i_layer])
        for j_node in range(sizes[i_layer+1]):
            node_bias = layer_biases[j_node]
            node_weights = list(layer_weights[j_node])
            dot = 0.0
            for k in range(len(node_weights)):
                dot += node_weights[k] * a[k]
            z = dot + node_bias

            # sigmoid
            sig = 1.0/(1.0 + math.exp(-z))
            # print(sig)

            b.append(sig)

        a = b
    print(a)


def main2():

    n_epochs = 2
    network_sizes = [392, 100, 1393]

    # Read in out data for fitting
    raw_data = []

    with open("../train_1/data_23/all_data_transformed.csv") as f:
        n_line = 0
        for line in f:
            n_line += 1
            if n_line > 1:
                split_line = [float(s) for s in line.split(",")]

                hour_diff = int(split_line[1])
                if hour_diff < 1393:  # Skip early ones
                    raw_data.append((np.reshape(split_line[2:], (len(split_line[2:]), 1)),
                                     vectorized_result3(int(split_line[1]))))

    print("Read data point: {}".format(len(raw_data)))

    network = Network(network_sizes)

    # Start the fit
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = network.SGD(
        raw_data, n_epochs, 4, 0.1, 0.001,
        monitor_training_cost=True)

    # multiplier = np.reshape([1, 2160], (2, 1))
    # for x, y in raw_data:
    #     result = network.feedforward(x) * multiplier
    #     yy = y * multiplier
    #     print(result[0], result[1], yy[0], yy[1])

    # results = network.evaluate3(raw_data)
    # print(results)
    # for x, y in raw_data:
    #     result = network.feedforward(x)
    #     print(np.argmax(result), result[1183])


def main3():

    n_epochs = 1000
    network_sizes = [392, 30, 1]

    # Read in out data for fitting
    raw_data = []

    with open("../train_1/test/regression_test2.csv") as f:
        n_line = 0
        for line in f:
            n_line += 1
            if n_line > 1:
                split_line = [float(s) for s in line.split(",")]
                raw_data.append((np.reshape(split_line[2:], (len(split_line[2:]), 1)),
                                 vectorized_result4(int(split_line[1]))))

    print("Read data point: {}".format(len(raw_data)))

    network = Network(network_sizes)

    # Start the fit
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = network.SGD(
        raw_data, n_epochs, 4, 0.1, 0.001,
        monitor_training_cost=True)

    multiplier = np.reshape([MAX_REGRESSION_VALUE], (1, 1))
    for x, y in raw_data:
        result = network.feedforward(x) * multiplier
        yy = y * multiplier
        print(result[0], yy[0])

    # results = network.evaluate3(raw_data)
    # print(results)
    # for x, y in raw_data:
    #     result = network.feedforward(x)
    #     print(np.argmax(result), result)


if __name__ == "__main__":
    main()
    # main2()
    # main3()
    # run()
    # test_network()
