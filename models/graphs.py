from autodiff.variables.variables import Tensor, TensorConst
from autodiff.operations.arithmetic import TensorSubtraction, TensorAddition, TensorMultiply, ElemwiseMultiply
from autodiff.operations.elements import TensorSquared, TensorSum

from .optimizers import OPTIMIZERS
from .losses import LOSSES


class NeuralNetwork:
    layers = []
    response = None
    setup = False
    vars = {}
    l2 = 0

    def fit(self, x, y, batch_size=1, epochs=1, optimizer='sgd', loss='mse', learning_rate=.001, l2=0):
        if not self.setup:
            self._setup_layers(Tensor(x[0]))
            self.setup = True
        self.l2 = l2
        def loss_fn(y, y_hat):
            ls = LOSSES[loss](y, y_hat)
            e = TensorAddition(ls, self.l2_loss())
            return e
        optimizer = OPTIMIZERS[optimizer](loss_fn, learning_rate)
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

    def l2_loss(self):
        weights_mag = None
        for var in self.vars:
            weights = TensorSum(TensorSquared(self.vars[var]))
            if weights_mag:
                weights_mag = TensorAddition(weights_mag, weights)
            else:
                weights_mag = weights
        return ElemwiseMultiply(TensorConst([self.l2]), weights_mag)

    def _setup_layers(self, x):
        self.vars = {}
        for layer in self.layers:
            layer.setup(x)
            x = layer.feed(x)
            for var_uid, var in layer.variables.items():
                self.vars[var_uid] = var
