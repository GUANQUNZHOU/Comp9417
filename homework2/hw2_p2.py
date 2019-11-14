import time
import numpy

# helper functions to read data


def read_data(file_name, encoding_function, expected_length):
    with open(file_name, 'r') as f:
        return numpy.array([encoding_function(row) for row in f.read().split('\n') if len(row) == expected_length])


def encode_string(s):
    return [one_hot_encode_character(c) for c in s]


def one_hot_encode_character(c):
    base = [0] * 26
    index = ord(c) - ord('a')
    if index >= 0 and index <= 25:
        base[index] = 1
    return base


def reverse_one_hot_encode(v):
    return chr(numpy.argmax(v) + ord('a')) if max(v) > 0 else ' '

# functions used in the neural network


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def sigmoid_derivative(x):
    return (1 - x) * x


def argmax(x):
    return numpy.argmax(x, axis=1)


class NeuralNetwork:
    def __init__(self, learning_rate=2, epochs=5000, input_size=9, hidden_layer_size=64):
        # activation function and its derivative to be used in backpropagation
        self.activation_function = sigmoid
        self.derivative_of_activation_function = sigmoid_derivative
        self.map_output_to_prediction = argmax

        # parameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size

        # initialisation
        numpy.random.seed(77)

    def fit(self, X, y):
        # reset timer
        timer_base = time.time()

        # initialise the weights of the NN
        input_dim = X.shape[2] + self.hidden_layer_size
        output_dim = y.shape[1]

        self.weights, self.biases = [], []

        previous_layer_size = input_dim
        for current_layer_size in [self.hidden_layer_size, output_dim]:
            # random initial weights and zero biases
            weights_of_current_layer = numpy.random.randn(
                previous_layer_size, current_layer_size)
            bias_of_current_layer = numpy.zeros((1, current_layer_size))

            self.weights.append(weights_of_current_layer)
            self.biases.append(bias_of_current_layer)
            previous_layer_size = current_layer_size

        # train the NN
        self.accuracy_log = []

        for epoch in range(self.epochs + 1):
            outputs = self.forward_propagate(X)
            prediction = outputs.pop()

            if epoch % 100 == 0:
                accuracy = self.evaluate(prediction, y)
                print(
                    f"In iteration {epoch}, training accuracy is {accuracy}.")
                self.accuracy_log.append(accuracy)

            # first step of back-propagation
            dEdz = y - prediction
            layer_input = outputs.pop()
            layer_output = prediction

            # calculate gradients
            dEds, dW, db = self.derivatives_of_last_layer(
                dEdz, layer_output, layer_input)

            # update weights
            self.weights[1] += self.learning_rate / X.shape[0] * dW
            self.biases[1] += self.learning_rate / X.shape[0] * db
            # print('ii', self.weights[1].shape)
            # setup for the next step
            previous_gradients = dEds
            layer_output = layer_input

            # back-propagation through time (unrolled)
            for step in range(self.input_size - 1, -1, -1):
                # TO DO: setup for the current step
                if step != 0:
                    out = outputs.pop()
                else:
                    out = numpy.zeros((X.shape[0], self.hidden_layer_size))
                if step == self.input_size - 1:
                    layer_input = numpy.hstack((out,X[:, step, :]))
                    weight = self.weights[1]
                else:
                    layer_input = numpy.hstack((out,X[:, step, :]))
                    weight = self.weights[0]
                    weight = weight[:64,:]
            
                # TO DO: calculate gradients
                gradients, dW, db = self.derivatives_of_hidden_layer(
                    previous_gradients, layer_output, layer_input, weight)

                self.weights[0] += self.learning_rate / X.shape[0] * dW
                self.biases[0] += self.learning_rate / X.shape[0] * db

                # TO DO: setup for the next step
                previous_gradients = gradients
                layer_output = out

        print(f"Finished training in {time.time() - timer_base} seconds")

    def test(self, X, y, verbose=True):
        predictions = self.forward_propagate(X)[-1]

        if verbose:
            for index in range(len(predictions)):
                prefix = ''.join(reverse_one_hot_encode(v) for v in X[index])
                print(
                    f"Expected {prefix + reverse_one_hot_encode(y[index])}, predicted {prefix + reverse_one_hot_encode(predictions[index])}")

        print(f"Testing accuracy: {self.evaluate(predictions, y)}")

    def evaluate(self, predictions, target_values):
        successful_predictions = numpy.where(self.map_output_to_prediction(
            predictions) == self.map_output_to_prediction(target_values))
        return successful_predictions[0].shape[0] / len(predictions) if successful_predictions else 0

    def forward_propagate(self, X):
        # initial states
        current_state = numpy.zeros((X.shape[0], self.hidden_layer_size))
        outputs = [current_state]

        # forward propagation through time (unrolled)
        for step in range(self.input_size):
            x = numpy.concatenate((current_state, X[:, step, :]), axis=1)
            current_state = self.apply_neuron(
                self.weights[0], self.biases[0], x)
            outputs.append(current_state)

        # the last layer
        output = self.apply_neuron(
            self.weights[1], self.biases[1], current_state)
        outputs.append(output)
        return outputs

    def apply_neuron(self, w, b, x):
        return self.activation_function(numpy.dot(x, w) + b)

    def derivatives_of_last_layer(self, dEdz, layer_output, layer_input):
        dEds = self.derivative_of_activation_function(layer_output) * dEdz
        dW = numpy.dot(layer_input.T, dEds)
        db = numpy.sum(dEds, axis=0, keepdims=True)# one single number 
        return dEds, dW, db

    def derivatives_of_hidden_layer(self, layer_difference, layer_output, layer_input, weight):
        # print('difference', layer_difference.shape)
        gradients = self.derivative_of_activation_function(
            layer_output) * numpy.dot(layer_difference, weight.T)
        dW = numpy.dot(layer_input.T, gradients)
        db = numpy.sum(gradients, axis=0, keepdims=True)
        return gradients, dW, db


training_input = read_data("training_input.txt", encode_string, 9)
training_label = read_data("training_label.txt", one_hot_encode_character, 1)

model = NeuralNetwork()
model.fit(training_input, training_label)

testing_input = read_data("testing_input.txt", encode_string, 9)
testing_label = read_data("testing_label.txt", one_hot_encode_character, 1)

model.test(testing_input, testing_label, verbose=True)
