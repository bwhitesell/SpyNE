import numpy as np

from .base import DualTensorOperation


class TensorAddition(DualTensorOperation):
    name = 'Tensor Addition'

    def execute(self):
        return np.add(self._a, self._b)

    @staticmethod
    def vector_jacobian_product(func):
        def a_vjp(g):
            g = func(g)
            return g

        def b_vjp(g):
            g = func(g)
            return g

        return a_vjp, b_vjp


class TensorSubtraction(DualTensorOperation):
    name = 'Tensor Subtraction'

    def execute(self):
        return np.add(self._a, self._b)

    @staticmethod
    def vector_jacobian_product(func):
        def a_vjp(g):
            g = func(g)
            return -1 * g

        def b_vjp(g):
            g = func(g)
            return -1 * g

        return a_vjp, b_vjp


class TensorMultiply(DualTensorOperation): 
    name = 'Tensor Multiplication'

    def execute(self):
        return np.dot(self._a, self._b)

    def vector_jacobian_product(self, func):
        a = self._a
        b = self._b


        if len(self._b.shape) >= 2:
          def a_vjp(g):
              g = func(g)
              return np.dot(g, np.swapaxes(b, -1, -2))

        elif len(self._b.shape) == 1 and len(self._a.shape) >= 2:
            def a_vjp(g):
                g = func(g)
                return np.outer(g, b)
        elif len(self._b.shape) == 1 and len(self._a.shape) == 1:
            def a_vjp(g):
                g = func(g)
                return g * b

        if len(self._a.shape) >= 2:
            def b_vjp(g):
                g = func(g)
                return np.dot(np.swapaxes(a, -1, -2), g)

        elif len(self._a.shape) == 1 and len(self._b.shape) >= 2:
            def b_vjp(g):
                g = func(g)
                return np.outer(a, g)

        elif len(self._a.shape) == 1 and len(self._b.shape) == 1:
            def b_vjp(g):
                g = func(g)
                return g * a

        return a_vjp, b_vjp
