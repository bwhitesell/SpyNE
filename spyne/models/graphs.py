from spyne.data_structures import Tensor, TensorConst
from spyne.operations import TensorAddition, TensorElemMultiply
from spyne.operations import TensorSquared, TensorSum

from .optimizers import OPTIMIZERS
from .losses import LOSSES


class NeuralNetwork:
    layers = []
    response = None
    setup = False
    vars = {}
    l2 = 0

    def fit(self, x, y, batch_size=1, epochs=1, optimizer='sgd', loss='mse', learning_rate=.01, l2=0,
            early_stopping=True):
        if not self.setup:
            self._setup_layers(Tensor(x[0]))
            self.setup = True
        self.l2 = l2
        optimizer = OPTIMIZERS[optimizer](self._build_loss_function(loss), learning_rate)
        optimizer.optimize(self, x, y, batch_size, epochs, early_stopping=early_stopping)

    def predict(self, x):
        return [self.forward_pass(Tensor(i)) for i in x]

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_pass(self, x):
        impulse = x
        for layer in self.layers:
            impulse = layer.feed(impulse)
        return impulse

    def l2_loss(self):
        weights_mag = None
        for var in self.vars:
            weights = TensorSum(TensorSquared(self.vars[var]))
            if weights_mag:
                weights_mag = TensorAddition(weights_mag, weights)
            else:
                weights_mag = weights
        return TensorElemMultiply(TensorConst([self.l2]), weights_mag)

    @property
    def n_layers(self):
        return len(self.layers)

    @property
    def n_parameters(self):
        n_params = 0
        if not self.setup:
            raise AttributeError('NN has not been setup yet. Try setting up with model with the fit method first.')
        for var in self.vars:
            n_params += self.vars[var].value.size
        return n_params

    def _build_loss_function(self, loss):
        def loss_function(y, y_hat):
            spec_loss = LOSSES[loss](y, y_hat)

            return spec_loss
        return loss_function

    def _setup_layers(self, x):
        self.vars = {}
        for layer in self.layers:
            layer.setup(x)
            x = layer.feed(x)
            for var_uid, var in layer.variables.items():
                self.vars[var_uid] = var
