import numpy as np

from autodiff.operations.base import DualTensorOperation
from autodiff.operations.utils import nest_func


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

        if len(self._b.shape) >= 2:
            @nest_func(func)
            def a_vjp(g):
                return np.dot(g, np.swapaxes(b, -1, -2))

        elif len(self._b.shape) == 1 and len(self._a.shape) >= 2:
            @nest_func(func)
            def a_vjp(g):
                return np.outer(g, b)
        elif len(self._b.shape) == 1 and len(self._a.shape) == 1:
            @nest_func(func)
            def a_vjp(g):
                return g * b

        if len(self._a.shape) >= 2:
            @nest_func(func)
            def b_vjp(g):
                return np.dot(np.swapaxes(a, -1, -2), g)

        elif len(self._a.shape) == 1 and len(self._b.shape) >= 2:
            @nest_func(func)
            def b_vjp(g):
                return np.outer(a, g)

        elif len(self._a.shape) == 1 and len(self._b.shape) == 1:
            @nest_func(func)
            def b_vjp(g):
                return g * a

        return a_vjp, b_vjp


class ElemwiseMultiply(DualTensorOperation):
    name = 'ElemWise Multiplication'

    def execute(self):
        return np.multiply(self._a, self._b)[0]

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
