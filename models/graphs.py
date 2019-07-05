from autodiff.variables.variables import Tensor

from .optimizers import OPTIMIZERS


class NeuralNetwork:
    layers = []
    response = None
    setup = False
    vars = {}

    def fit(self, x, y, batch_size=1, epochs=1, optimizer='sgd', loss='mse', learning_rate=.001):
        if not self.setup:
            self._setup_layers(Tensor(x[0]))
            self.setup = True
        optimizer = OPTIMIZERS[optimizer](loss, learning_rate)
        optimizer.optimize(self, x, y, batch_size, epochs)

    def predict(self, x):
        return self.forward_pass(self, Tensor(x))

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
