import numpy as np

from spyne.operations import TensorMultiply, TensorAddition, TensorDuplicateRows
from spyne.tensors import Tensor

from .constants import ACTIVATIONS, XAVIER_INIT_PARAM


class FullyConnectedLayer:
    """ A fully connected layer of a multi-layer perceptron. """
    
    w = m = b = _b = z = a = None
    input_shape = w_shape = None
    variables = {}
    weights_shape = products_shape = ()

    def __init__(self, neurons, activation='relu', dropout=0):
        self.neurons = neurons
        self.dropout = dropout
        self.activation_key = activation

    def setup(self, x):
        self.variables = {}
        self.input_shape = x.shape
        self.weights_shape = self._get_weights_shape(x)
        self.products_shape = self._get_product_shape(x)
        self._process_activation()

        # xavier initialization
        self.w = self._add_var(np.random.uniform(-self.r, self.r, self.weights_shape))
        self.b = self._add_var(np.random.random((self.neurons,)))

    def feed(self, x):
        self._check_input(x)
        self.m = TensorMultiply(x, self.w)
        self._b = TensorDuplicateRows(self.b, x.shape[0])
        self.z = TensorAddition(self.m, self._b)
        self.a = self.activation(self.z)
        if self.dropout > 0:
            self.a.value = self._dropout(self.a.value)
        return self.a

    def _dropout(self, a):
        drops = np.random.binomial(1, self.dropout, self.a.shape)
        return drops * a

    def _get_weights_shape(self, x):
        return x.shape[1], self.neurons

    def _get_product_shape(self, x):
        w_shape = self._get_weights_shape(x)
        x_shape = x.shape if len(x.shape) > 1 else (1,) + x.shape
        if w_shape[len(w_shape)-1] == 1 and x_shape[len(x_shape)-2] == 1:
            return 1,

        else:
            x_comp = x_shape[:len(x_shape)-2]
            w_comp = w_shape[:len(w_shape) - 1] + \
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
        if x.shape[1] != self.input_shape[1]:
            raise ValueError(
                f'''This layer is constructed to handle inputs of shape {self.input_shape},
                    not of shape {x.shape}'''
            )

    def _process_activation(self):
        self.activation = ACTIVATIONS[self.activation_key]
        r = np.sqrt(6 / (self.input_shape[1] + self.neurons))
        self.r = XAVIER_INIT_PARAM[self.activation_key] * r
