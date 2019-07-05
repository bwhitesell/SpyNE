import numpy as np
from .constants import ACTIVATIONS


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