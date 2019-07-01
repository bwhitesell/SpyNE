import numpy as np

from .base import DualTensorOperation


class TensorAddition(DualTensorOperation):
    name = 'Tensor Addition'

    def execute(self):
        return np.add(self._a, self._b)

    @staticmethod
    def vector_jacobian_product():
        def a_vjp(g):
            return g

        def b_vjp(g):
            return g

        return a_vjp, b_vjp


class TensorSubtraction(DualTensorOperation):
    name = 'Tensor Subtraction'

    def execute(self):
        return np.add(self._a, self._b)

    @staticmethod
    def vector_jacobian_product():
        def a_vjp(g):
            return -1 * g

        def b_vjp(g):
            return -1 * g

        return a_vjp, b_vjp


class TensorMultiply(DualTensorOperation): 
    name = 'Tensor Multiplication'

    def execute(self):
        return np.dot(self._a, self._b)

    def vector_jacobian_product(self):
        """ Returns the VJPs for the Jacobians of the arguments. """
        a = self._a
        b = self._b

        if len(self._a.shape) >= 1 and len(self._b.shape) >= 1:
            def a_vjp(g):
                """ g dotted with Jacobian of operation with respect to A"""
                return np.dot(g, np.swapaxes(b, -1, -2))

            def b_vjp(g):
                """ g dotted with Jacobian of operation with respect to B """
                return np.dot(np.swapaxes(a, -1, -2), g)

        else:
            raise NotImplementedError('''the VJP builder for TensorMultiply can only handle 
                                      non-scalar arguments for now.''')

        return a_vjp, b_vjp
