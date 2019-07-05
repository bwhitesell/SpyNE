from autodiff.differentiation.derivatives import BackwardsPass
from autodiff.operations.arithmetic import TensorSubtraction
from autodiff.operations.elements import TensorSquared, TensorSum
from autodiff.variables.variables import Tensor


class BaseOptimizer:
    loss = None

    def __init__(self, loss):
        self.loss = loss

    def optimize(self, nn, x, y, batch_size, epochs):
        n_batches = int(x.shape[0] / batch_size)
        for epoch in range(epochs):
            for batch in range(n_batches):
                batch_grad = {}
                batch_loss = 0
                start_splice = batch * batch_size
                end_splice = (batch + 1) * batch_size

                x_batch = x[start_splice:end_splice, ...]
                y_batch = y[start_splice:end_splice, ...]

                for elem in range(batch_size):
                    # forward pass
                    y_hat = nn.forward_pass(Tensor(x_batch[elem, ...]))
                    loss = self.loss(Tensor(y_batch[elem, ...]), y_hat)
                    batch_loss += loss.value
                    # backwards pass
                    grad = BackwardsPass(loss).jacobians()
                    for var in grad:
                        if var not in batch_grad:
                            batch_grad[var] = grad[var]
                        else:
                            batch_grad[var] += grad[var]

                for var in grad:
                    grad[var] = grad[var] / batch_size

                self._update(nn, grad)
                print(f'Batch: {batch}')
                print(f'Batch Loss: {batch_loss / batch_size}')

    def _update(self, nn, grad):
        pass


class SGDOptimizer:
    updates = {}
    learning_rate = .001

    def __init__(self, learning_rate, *args):
        super().__init__(*args)
        self.learning_rate = learning_rate

    def _update(self, nn, gradient):
        for var_uid in nn.vars:
            nn.vars[var_uid].value -= gradient[var_uid] * self.learning_rate


class MomentumOptimizer:
    pass


class RMSPropOptimizer:
    pass


class AdamOptimizer:
    pass


OPTIMIZERS = {
    'sgd': SGDOptimizer,
    'momentum': MomentumOptimizer,
    'rmsprop': RMSPropOptimizer,
    'adam': AdamOptimizer,
}
