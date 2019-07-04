import numpy as np

from autodiff.differentiation.derivatives import BackwardsPass
from autodiff.operations.arithmetic import TensorMultiply, TensorAddition, TensorSubtraction
from autodiff.operations.elements import TensorSquared, TensorSum
from autodiff.variables.variables import Tensor

from .constants import ACTIVATIONS


class NeuralNetwork:
    layers = []
    response = None
    loss = 'mse'
    learning_rate = .0001

    def add_layer(self, layer):
        self.layers.append(layer)

    def fit(self, x, y, batch_size=1):
        self._setup_layers(x)
        for iter in range(batch_size):
            self._learn_iter(x[iter, ...], y[iter, ...])

    def _setup_layers(self, x):
        for layer in self.layers:
            x = layer.setup(x)

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

    def _learn_iter(self, x, y):
        y_hat = self._forward_pass(x)
        loss = self.loss_function(y, y_hat)
        return BackwardsPass(loss).jacobians()




class FullyConnectedLayer:
    variable_nodes = (w, b)
    w = m = b = z = a = None
    input_shape = w_shape = None

    def __init__(self, x, neurons, activation='relu'):
        self.neurons = neurons
        self.activation = ACTIVATIONS[activation]

    def setup(self, x):
        self.input_shape = x.shape
        self.w_shape = self._get_weights_shape(x)
        self.w = Tensor(np.random.normal(0, 1, self.w_shape))
        self.b = Tensor(np.random.random(self._get_product_shape()))
        self.record_nodes()

        return self.feed()

    def feed(self, x):
        self._check_input(x)
        self.m = TensorMultiply(x, self.w)
        self.z = TensorAddition(self.m, self.b)
        self.a = self.activation(self.z)
        return self.a

    def record_nodes(self):
        self.nodes = {
            self.w.node_uid: self.w
        }

    def _get_weights_shape(self, x):
        n_dims = len(x.shape)
        w_shape = list(x.shape)
        w_shape[n_dims - 2] = x.shape[n_dims - 1]
        w_shape.append(self.neurons)
        return tuple(w_shape)

    def _get_product_shape(self):
        w_comp = self.w_shape[:len(self.w_shape)-2]
        if len(self.input_shape) > 1:
            x_comp = self.input_shape[:len(self.input_shape) - 2] + \
                self.input_shape[len(self.input_shape) - 1:]
        else:
            x_comp = (1,)
        return w_comp + x_comp

    def _check_input(self, x):
        """Give a user clear feedback if there is a shape mismatch."""
        if x.shape != self.input_shape:
            raise ValueError(
                f'''This layer is constructed to handle inputs of shape {self.input_shape},
                    not of shape {x.shape}'''
            )


def activations(x):
    if x == 'ReLU':
        return TensorReLU



