import numpy as np

from .base import UniTensorOperation


class TensorReLU(UniTensorOperation):
    name = 'Tensor ReLU'

    def execute(self):
        return self._a * (self._a > 0)

    def vector_jacobian_product(self, g):
        return g * np.where(self._a > 0, 1, 0)


class TensorTanh(UniTensorOperation):
    name = 'Tensor Tanh'

    def execute(self):
        return np.tanh(self._a)

    def vector_jacobian_product(self, g):
        return g * 1 / np.square(np.cosh(self._a))


class TensorSigmoid(UniTensorOperation):
    name = 'Tensor Sigmoid'

    def execute(self):
        e = np.exp(self._a)
        return e / (e + 1)

    def vector_jacobian_product(self, g):
        e = np.exp(self._a)
        return g * (e / (e**2 + 1))
