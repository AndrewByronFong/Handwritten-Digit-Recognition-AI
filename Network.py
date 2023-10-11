import numpy as np
import mnist_loader as ml


class Network:
    def __init__(self, shape):
        self.shape = shape
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(shape[:-1], shape[1:])]
        self.biases = [np.random.randn(y,1) for y in shape[1:]]
        self.num_layers = len(shape)

    def SGD(self, eta, epochs, mini_batch_size):
        training_data, validation_data, test_data = ml.load_data_wrapper(
            mini_batch_size)
        training_data = list(training_data)
        test_data = list(test_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            for x, y in training_data:
                activations, z_value = self.feedforward(x)
                delta = cost_derivative(
                    activations[self.num_layers-1], y) * sigmoid_prime(z_value[self.num_layers-1])
                nabla_b = np.empty((0, 1))
                nabla_b = np.array([sum(nb) * eta for nb in delta])
                nabla_b = nabla_b.reshape(len(nabla_b),1) / mini_batch_size
                nabla_w = np.dot(
                    delta, activations[self.num_layers-2].transpose()) * eta / mini_batch_size
                self.weights[self.num_layers-2] -= nabla_w
                self.biases[self.num_layers-2] -= nabla_b
                for l in range(self.num_layers-2, 0, -1):
                    delta = np.dot(
                        self.weights[l].transpose(), delta) * sigmoid_prime(z_value[l])
                    nabla_b = np.array([sum(nb) * eta for nb in delta])
                    nabla_b = nabla_b.reshape(len(nabla_b),1) / mini_batch_size
                    nabla_w = np.dot(delta, activations[l-1].transpose()) * eta / mini_batch_size
                    self.weights[l-1] -= nabla_w
                    self.biases[l-1] -= nabla_b
            print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),len(test_data)))
    
    def evaluate(self, test_data):
        test_results =[]
        for(x, y) in test_data:
            activations, z_value = self.feedforward(x)
            test_results.append((np.argmax(activations[self.num_layers-1]), y))
        return sum(int(x == y) for (x, y) in test_results)

    def feedforward(self, x):
        activations = [x]
        z_value = [np.empty(0)]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[len(activations)-1]) + b
            z_value.append(z)
            activations.append(sigmoid(z))
        return activations, z_value


def cost_derivative(output_matrix, y):
    return (output_matrix - y)


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
