import numpy as np

from .base import UniTensorOperation
from variables import Tensor


class TensorSum(UniTensorOperation):
    name = 'Tensor Sum'

    def __init__(self, a):
        self.a = a
        self._check_types()
        self.value = self.execute()

    def execute(self):
        return Tensor(self.a.sum())

    @staticmethod
    def vector_jacobian_product():
        def a_vjp(g):
            return g.sum()
        return a_vjp


class TensorSquared(UniTensorOperation):
    name = 'Tensor Square'

    def __init__(self, a):
        self.a = a
        self._check_types()
        self.value = self.execute()

    def execute(self):
        return Tensor(np.square(self.a))

    def vector_jacobian_product(self):
        def a_vjp(g):
            return g * 2 * self.a
        return a_vjp