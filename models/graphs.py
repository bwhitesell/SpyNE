from autodiff.differentiation.derivatives import BackwardsPass
from autodiff.operations.arithmetic import TensorSubtraction
from autodiff.operations.elements import TensorSquared, TensorSum
from autodiff.variables.variables import Tensor

from .optimizers import OPTIMIZERS
from .losses import LOSSES


class NeuralNetwork:
    layers = []
    response = None
    setup = False
    vars = {}

    def fit(self, x, y, batch_size=1, epochs=1, optimizer='sgd', loss='mse'):
        if not self.setup:
            self._setup_layers(Tensor(x[0]))
            self.setup = True
        loss_func = LOSSES[loss]
        optimizer = OPTIMIZERS[optimizer](loss_func)
        optimizer.optimize(self, x, y, batch_size, epochs)

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_pass(self, x):
        impulse = x
        for layer in self.layers:
            impulse = layer.feed(impulse)
        return impulse

    def _setup_layers(self, x):
        self.vars = {}
        for layer in self.layers:
            layer.setup(x)
            x = layer.feed(x)
            for var_uid, var in layer.variables.items():
                self.vars[var_uid] = var

    def _update(self, gradient):
        for var_uid in self.vars:
            self.vars[var_uid].value -= gradient[var_uid] * self.learning_rate
