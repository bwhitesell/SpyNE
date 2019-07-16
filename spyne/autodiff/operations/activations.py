import numpy as np

from .base import UniTensorOperation
from .utils import nest_func


class TensorReLU(UniTensorOperation):
    name = 'Tensor ReLU'

    def execute(self):
        return self._a * (self._a > 0)

    def vector_jacobian_product(self, func=lambda g: g):
        a = self._a

        @nest_func(func)
        def a_vjp(g):
            return g * np.where(a > 0, 1, 0)

        return a_vjp


class TensorTanh(UniTensorOperation):
    name = 'Tensor Tanh'

    def execute(self):
        return np.tanh(self._a)

    def vector_jacobian_product(self, func=lambda g: g):
        a = self._a

        @nest_func(func)
        def a_vjp(g):
            return g * 1 / np.square(np.cosh(a))

        return a_vjp


class TensorSigmoid(UniTensorOperation):
    name = 'Tensor Sigmoid'

    def execute(self):
        e = np.exp(self._a)
        return e / (e + 1)

    def vector_jacobian_product(self, func=lambda g: g):
        e = np.exp(self._a)

        @nest_func(func)
        def a_vjp(g):
            return np.multiply(g, np.divide(e, np.square((1 + e))))

        return a_vjp


class TensorSoftmax(UniTensorOperation):
    name = 'Tensor Softmax'

    def execute(self):
        exp_sum = np.exp(self._a).sum()
        return np.exp(self._a) / exp_sum

    def vector_jacobian_product(self, func=lambda g: g):
        a = self._a

        @nest_func(func)
        def a_vjp(g):
            exp_sum = np.exp(a).sum()
            num = np.multiply(np.exp(a), exp_sum) - np.exp(2 * a)
            return np.multiply(g, num / np.square(exp_sum))

        return a_vjp

