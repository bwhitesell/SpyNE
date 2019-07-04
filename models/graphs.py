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
    learning_rate = .00001
    vars = {}

    def add_layer(self, layer):
        self.layers.append(layer)

    def fit(self, x, y, batch_size=1):
        self._setup_layers(x)
        batch_grad = {}

        for itr in range(batch_size):
            grad = self._learn_iter(x[itr, ...], y[itr, ...])
            for var in grad:
                batch_grad[var] += grad[var]

        for var in grad:
            grad[var] = grad[var] / batch_size

    def _setup_layers(self, x):
        self.vars = {}
        for layer in self.layers:
            layer.setup(x)
            x = layer.feed(x)
            for var_uid, var in layer.variables.items():
                self.vars[var_uid] = var

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
        gradient = BackwardsPass(loss).jacobians()
        self._update(gradient)
        print(y_hat)

    def _update(self, gradient):
        for var_uid in gradient:
            self.vars[var_uid].value += gradient[var_uid] * self.learning_rate


class FullyConnectedLayer:
    w = m = b = z = a = None
    input_shape = w_shape = None
    variables = {}

    def __init__(self, neurons, activation='relu'):
        self.neurons = neurons
        self.activation = ACTIVATIONS[activation]

    def setup(self, x):
        self.variables = {}
        self.input_shape = x.shape

        self.w = self._add_var(np.random.normal(0, 1, self._get_weights_shape(x)))
        self.b = self._add_var(np.random.random(self._get_product_shape(x)))

    def feed(self, x):
        self._check_input(x)
        self.m = TensorMultiply(x, self.w)
        self.z = TensorAddition(self.m, self.b)
        self.a = self.activation(self.z)
        return self.a

    def _get_weights_shape(self, x):
        n_dims = len(x.shape)
        w_shape = list(x.shape)
        w_shape[n_dims - 2] = x.shape[n_dims - 1]
        w_shape.append(self.neurons)
        return tuple(w_shape)

    def _get_product_shape(self, x):
        w_shape = self._get_weights_shape(x)
        x_shape = x.shape if len(x.shape) > 1 else (1,) + x.shape
        if w_shape[len(w_shape)-1] == 1 and x_shape[len(x_shape)-2] == 1:
            return 1,

        else:
            x_comp = x_shape[:len(x_shape)-2]
            w_comp = w_shape[:len(w_shape) - 2] + \
                w_shape[len(w_shape) - 1:]
            return w_comp + x_comp

    @property
    def shape(self):
        if self.b:
            return self.b.shape
        else:
            raise AttributeError("Layer has no shape until the setup() method has been run.")

    def _add_var(self, init_data):
        var = Tensor(init_data)
        self.variables[var.node_uid] = var
        return var

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



