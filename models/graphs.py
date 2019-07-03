import numpy as np

from autodiff.differentiation.derivatives import BackwardsPass
from autodiff.operations.activations import TensorReLU
from autodiff.operations.arithmetic import TensorMultiply, TensorAddition, TensorSubtraction
from autodiff.operations.elements import TensorSquared, TensorSum
from autodiff.variables.variables import Tensor


class NeuralNetwork:
    layers = []
    response = None
    loss = 'mse'
    learning_rate = .0003

    def add_layer(self, layer):
        self.layers.append(layer)

    def loss_function(self, y_hat, y):
        error = TensorSubtraction(y, y_hat)
        if self.loss == 'mse':
            error_sq = TensorSquared(error)
        return TensorSum(error_sq)

    def _forward_pass(self, x):
        impulse = x
        for layer in self.layers:
            impulse = layer.feed(impulse)

        return impulse

    def _backward_pass(self, loss):
        self.gradient = BackwardsPass(loss).jacobians(update=True, alpha=self.learning_rate)

    def _learn_row(self, x, y):
        y_hat = self._forward_pass(x)
        loss = self.loss_function(y, y_hat)
        self._backward_pass(loss)
        print(y_hat)


class FullyConnectedLayer:
    w = m = b = z = a = None

    def __init__(self, x, neurons, activation='ReLU'):
        self.input_shape = x.shape
        self.neurons = neurons
        self.activation = activation
        self.w_shape = self._get_weights_shape(x)
        self.w = Tensor(np.random.normal(0, 1, self.w_shape))
        self.m = TensorMultiply(x, self.w)
        self.b = Tensor(np.random.random(self.m.shape))
        self.feed(x)

    def feed(self, x):
        self._check_input(x)
        self.m = TensorMultiply(x, self.w)
        self.z = TensorAddition(self.m, self.b)
        if self.activation == 'ReLU':
            self.a = TensorReLU(self.z)
        else:
            self.a = self.z
        return self.a

    def _get_weights_shape(self, x):
        n_dims = len(x.shape)
        w_shape = list(x.shape)
        w_shape[n_dims - 2] = x.shape[n_dims - 1]
        w_shape.append(self.neurons)
        return tuple(w_shape)

    def _check_input(self, x):
        """Give a user clear feedback if there is a shape mismatch."""
        if x.shape != self.input_shape:
            raise ValueError(
                f'''This layer is constructed to handle inputs of shape {self.input_shape}'''
            )


def activations(x):
    if x == 'ReLU':
        return TensorReLU



