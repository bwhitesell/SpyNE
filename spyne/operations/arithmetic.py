import numpy as np

from .base import DualTensorOperation
from .utils import nest_func


class TensorAddition(DualTensorOperation):
    name = 'Tensor Addition'

    def execute(self):
        return np.add(self._a, self._b)

    @staticmethod
    def vector_jacobian_product(func=lambda g: g):
        @nest_func(func)
        def a_vjp(g):
            return g

        @nest_func(func)
        def b_vjp(g):
            return g

        return a_vjp, b_vjp


class TensorSubtraction(DualTensorOperation):
    name = 'Tensor Subtraction'

    def execute(self):
        return np.subtract(self._a, self._b)

    @staticmethod
    def vector_jacobian_product(func=lambda g: g):

        @nest_func(func)
        def a_vjp(g):
            return -1 * g

        @nest_func(func)
        def b_vjp(g):
            return -1 * g

        return a_vjp, b_vjp


class TensorMultiply(DualTensorOperation): 
    name = 'Tensor Multiplication'

    def execute(self):
        return np.dot(self._a, self._b)

    def vector_jacobian_product(self, func=lambda g: g):
        a = self._a
        b = self._b

        a_ndim = len(a.shape)
        b_ndim = len(b.shape)

        needs_transpose = b_ndim > 1 and a_ndim > 0
        swap = (lambda x: np.swapaxes(x, -1, -2)) if needs_transpose else (lambda x: x)

        if b_ndim > 1 and a_ndim > 0:
            @nest_func(func)
            def a_vjp(g):
                return np.tensordot(g, np.swapaxes(b, -1, -2), b_ndim - 1)

        else:
            @nest_func(func)
            def a_vjp(g):
                contract_num = max(0, len(b.shape) - (len(a.shape) != 0))
                return np.tensordot(g, b, contract_num)

        if a_ndim > 1 and b_ndim > 0:
            @nest_func(func)
            def b_vjp(g):
                out = swap(np.tensordot(
                    g, a, [range(-a_ndim - b_ndim + 2, -b_ndim + 1), range(a_ndim - 1)]))
                return out

        else:
            @nest_func(func)
            def b_vjp(g):
                contract_num = max(0, a_ndim - (b_ndim != 0))
                return swap(np.tensordot(g, a, contract_num))

        return a_vjp, b_vjp


class TensorElemMultiply(DualTensorOperation):
    name = 'ElemWise Multiplication'

    def execute(self):
        return np.multiply(self._a, self._b)

    def vector_jacobian_product(self, func=lambda g: g):
        a = self._a
        b = self._b

        @nest_func(func)
        def a_vjp(g):
            return np.multiply(g, b)

        @nest_func(func)
        def b_vjp(g):
            return np.multiply(g, a)

        return a_vjp, b_vjp
