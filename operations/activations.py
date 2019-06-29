import numpy as np

from .base import UniTensorOperation
from variables import Tensor


class TensorReLU(UniTensorOperation):
    name = 'Tensor ReLU'

    def __init__(self, a):
        self.a = a
        self._check_types()
        self.value = self.execute()

    def execute(self):
        return Tensor(self.a * (self.a > 0))

    def vector_jacobian_product(self):
        def a_vjp(g):
           return g * np.where(self.a > 0, 1, 0)
        return a_vjp


class TensorTanh(UniTensorOperation):
    name = 'Tensor Tanh'

    def __init__(self, a):
        self.a = a
        self._check_types()
        self.value = self.execute()

    def execute(self):
        return Tensor(np.tanh(self.a))

    def vector_jacobian_product(self):
        def a_vjp(g):
            return g * 1 / np.square(np.cosh(self.a))
        return a_vjp


class TensorSigmoid(UniTensorOperation):
    name = 'Tensor Sigmoid'

    def __init__(self, a):
        self.a = a
        self._check_types()
        self.value = self.execute()

    def execute(self):
        e = np.exp(self.a)
        return Tensor(e / (e + 1))

    def vector_jacobian_product(self):
        def a_vjp(g):
            e = np.exp(self.a)
            return g * (e / (e**2 + 1))

        return a_vjp