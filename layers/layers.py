import numpy as np

from autodiff.operations.arithmetic import TensorMultiply, TensorAddition
from autodiff.variables.variables import Tensor

from .constants import ACTIVATIONS


class FullyConnectedLayer:
    w = m = b = z = a = None
    input_shape = w_shape = None
    variables = {}
    weights_shape = products_shape = ()

    def __init__(self, neurons, activation='relu', dropout=0):
        self.neurons = neurons
        self.activation = ACTIVATIONS[activation]
        self.dropout = dropout

    def setup(self, x):
        self.variables = {}
        self.input_shape = x.shape
        self.weights_shape = self._get_weights_shape(x)
        self.products_shape = self._get_product_shape(x)

        # xavier initialization
        r = np.sqrt(6/(sum(self.input_shape) + sum(self.products_shape)))
        self.w = self._add_var(np.random.uniform(-r, r, self.weights_shape))
        self.b = self._add_var(np.random.random(self.products_shape))

    def feed(self, x):
        self._check_input(x)
        self.m = TensorMultiply(x, self.w)
        self.z = TensorAddition(self.m, self.b)
        self.a = self.activation(self.z)
        if self.dropout > 0:
            self.a.value = self._dropout(self.a.value)
        return self.a

    def _dropout(self, a):
        n_drops = np.random.binomial(a.size, self.dropout)
        indices = np.random.choice(np.arange(a.shape[0]), replace=False,
                                   size=n_drops)
        a[indices] = 0
        return a

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