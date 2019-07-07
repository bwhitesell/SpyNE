import numpy as np

from SpyNE.autodiff.differentiation.derivatives import BackwardsPass
from SpyNE.autodiff.variables.variables import Tensor


class BaseOptimizer:
    name = 'Base Optimizer'
    loss = None

    def __init__(self, loss, learning_rate):
        self.loss = loss
        self.learning_rate = learning_rate
        self.print_iter = 0

    def optimize(self, nn, x, y, batch_size, epochs, early_stopping):
        self._print_optimization_message(nn)
        x_train, x_test, y_train, y_test = self._train_val_split(x, y)

        n_batches = int(x_train.shape[0] / batch_size)
        for epoch in range(epochs):
            x_train, y_train = self._shuffle(x_train, y_train)
            for batch in range(n_batches):
                batch_grad = {}
                start_splice = batch * batch_size
                end_splice = (batch + 1) * batch_size

                x_batch = x_train[start_splice:end_splice, ...]
                y_batch = y_train[start_splice:end_splice, ...]

                for elem in range(batch_size):
                    # forward pass
                    y_hat = nn.forward_pass(Tensor(x_batch[elem, ...]))
                    loss = self.loss(Tensor(y_batch[elem, ...]), y_hat)

                    # backwards pass
                    grad = BackwardsPass(loss).jacobians()
                    for var in grad:
                        if var not in batch_grad:
                            batch_grad[var] = grad[var]
                        else:
                            batch_grad[var] += grad[var]

                for var in grad:
                    grad[var] = grad[var] / batch_size

                # update w/ optimization alg
                self._update(nn, grad)
                # control outputs
                self._handle_prints(epoch, batch, n_batches)
            # eval performance
            train_loss = self._eval_perf(x_train, y_train, nn)
            validation_loss = self._eval_perf(x_test, y_test, nn)

            # early stopping
            if early_stopping and epoch > 0:
                if validation_loss > lst_epch_val_loss:
                    self._handle_prints(epoch, batch, n_batches, train_loss, validation_loss)
                    break
            lst_epch_val_loss = validation_loss

            self._handle_prints(epoch, batch, n_batches, train_loss, validation_loss)

    @staticmethod
    def _train_val_split(x, y, split_size=.8):
        train_size = int(x.shape[0] * split_size)
        test_size = x.shape[0] - train_size

        x_train = x[:train_size]
        y_train = y[:train_size]
        x_test = x[-test_size:]
        y_test = y[-test_size:]
        return x_train, x_test, y_train, y_test

    @staticmethod
    def _shuffle(x, y):
        s = np.arange(x.shape[0])
        return x[s], y[s]

    def _eval_perf(self, x, y, model):
        n_evals = x.shape[0]
        loss = 0
        for t in range(n_evals):
            # forward pass
            y_hat = model.forward_pass(Tensor(x[t]))
            loss += self.loss(Tensor([y[t]]), y_hat).value
        return loss / n_evals

    def _print_optimization_message(self, nn):
        print('----------------------')
        print('Neural Network Architecture')
        print(f'Layers: {nn.n_layers}')
        print(f'Params: {nn.n_parameters} parameters')
        print(f'Using optimizer: {self.name}')
        print('\n')

    def _handle_prints(self, epoch, batch, n_batches, train_loss=0, val_loss=0):
        end = '\n' if (train_loss != 0 or val_loss != 0) else '\r'
        print(f'Batch {batch + 1}/{n_batches}, {round((batch + 1)/n_batches * 100, 4)}% for '
              + f'epoch {epoch}:  Train Loss: {round(train_loss, 4)} | Val Loss: {round(val_loss, 4)}', end=end)
        self.print_iter += 1

    def _update(self, nn, grad):
        pass


class SGDOptimizer(BaseOptimizer):
    name = 'Stochastic Gradient Descent Optimizer'

    def _update(self, nn, gradient):
        for var_uid in nn.vars:
            nn.vars[var_uid].value -= gradient[var_uid] * self.learning_rate


class MomentumOptimizer(BaseOptimizer):
    name = 'Momentum Optimizer'

    beta = 0
    m = {}

    def __init__(self, loss, learning_rate, beta=0.9):
        super().__init__(loss, learning_rate)
        self.beta = beta

    def _update(self, nn, gradient):
        for var_uid in nn.vars:
            if var_uid in self.m:
                self.m[var_uid] = self.beta * self.m[var_uid] - self.learning_rate * gradient[var_uid]
            else:
                self.m[var_uid] = -self.learning_rate * gradient[var_uid]

            nn.vars[var_uid].value = nn.vars[var_uid].value + self.m[var_uid]


class RMSPropOptimizer(BaseOptimizer):
    name = 'RMSProp Optimizer'

    m = {}

    def __init__(self, loss, learning_rate, beta=0.9, epsilon=1e-10):
        super().__init__(loss, learning_rate)
        self.beta = beta
        self.epsilon = epsilon

    def _update(self, nn, gradient):
        for var_uid in nn.vars:
            jv = self.learning_rate * gradient[var_uid]
            if var_uid in self.m:
                self.m[var_uid] = self.beta * self.m[var_uid] + (1 - self.beta) * (jv * jv)
            else:
                self.m[var_uid] = (1 - self.beta) * (jv * jv)

            nn.vars[var_uid].value += - self.learning_rate * jv / np.sqrt(self.m[var_uid] + self.epsilon)


OPTIMIZERS = {
    'sgd': SGDOptimizer,
    'momentum': MomentumOptimizer,
    'rmsprop': RMSPropOptimizer,
}
