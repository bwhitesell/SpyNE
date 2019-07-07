import numpy as np

from autodiff.differentiation.derivatives import BackwardsPass
from autodiff.variables.variables import Tensor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class BaseOptimizer:
    loss = None

    def __init__(self, loss, learning_rate):
        self.loss = loss
        self.learning_rate = learning_rate

    def optimize(self, nn, x, y, batch_size, epochs):
        x_train, x_test, y_train, y_test = self._train_val_split(x, y)

        n_batches = int(x_train.shape[0] / batch_size)
        for epoch in range(epochs):
            x_train, y_train = self._shuffle(x_train, y_train)
            print('-------------------')
            print(f'Epoch: {epoch}')
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

                self._update(nn, grad)
            lr = LinearRegression()
            lr.fit(x_train, y_train)
            y_lr_est = lr.predict(x_test)

            print(f'Epoch {epoch} linear_reg val-set loss: {mean_squared_error(y_test, y_lr_est)}')
            train_loss = self._eval_perf(x_train, y_train, nn)
            validation_loss = self._eval_perf(x_test, y_test, nn)
            print(f'Epoch {epoch} train-set loss: {train_loss}')
            print(f'Epoch {epoch} val-set loss: {validation_loss}')

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

    def _update(self, nn, grad):
        pass


class SGDOptimizer(BaseOptimizer):

    def _update(self, nn, gradient):
        for var_uid in nn.vars:
            nn.vars[var_uid].value -= gradient[var_uid] * self.learning_rate


class MomentumOptimizer(BaseOptimizer):
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


class AdamOptimizer(BaseOptimizer):
    m = {}
    s = {}

    def __init__(self, loss, learning_rate, beta_1=0.9,  beta_2=0.999, epsilon=1e-10):
        super().__init__(loss, learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter = 1
        self.m = {}
        self.s = {}

    def _update(self, nn, gradient):
        for var_uid in nn.vars:
            jv = self.learning_rate * gradient[var_uid]
            if var_uid in self.m:
                self.m[var_uid] = self.beta_1 * self.m[var_uid] - (1 - self.beta_1) * jv
            else:
                self.m[var_uid] = (1 - self.beta_1) * jv

            if var_uid in self.s:
                self.s[var_uid] = self.beta_2 * self.s[var_uid] + (1 - self.beta_2) * (jv * jv)
            else:
                self.s[var_uid] = (1 - self.beta_2) * (jv * jv)

            self.m[var_uid] = self.m[var_uid] / (1 - self.beta_1**self.n_iter)
            self.s[var_uid] = self.s[var_uid] / (1 - self.beta_2 ** self.n_iter)

            nn.vars[var_uid].value += np.divide(self.learning_rate * self.m[var_uid],np.sqrt(self.s[var_uid] + self.epsilon))



OPTIMIZERS = {
    'sgd': SGDOptimizer,
    'momentum': MomentumOptimizer,
    'rmsprop': RMSPropOptimizer,
    'adam': AdamOptimizer,
}
