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
        e = np.exp(-1 * self._a)

        @nest_func(func)
        def a_vjp(g):
            return np.multiply(g, e / (1 + e)**2)

        return a_vjp


class TensorSoftmax(UniTensorOperation):
    name = 'Tensor Softmax'

    def execute(self):
        return np.exp(self._a) / np.sum(np.exp(self._a), axis=1)[:, None]

    def vector_jacobian_product(self, func=lambda g: g):
        a = self._a

        @nest_func(func)
        def a_vjp(g):
            row_exp_sum = np.sum(np.exp(a), axis=1)
            elems = np.exp(a) * row_exp_sum[:, None] - np.square(np.exp(a))
            return np.multiply(g, elems)

        return a_vjp

