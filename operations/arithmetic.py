import numpy as np

from .base import DualTensorOperation


class TensorAddition(DualTensorOperation):
    name = 'Tensor Addition'

    def execute(self):
        return np.add(self._a, self._b)

    @staticmethod
    def vector_jacobian_product(g):
        return g, g


class TensorSubtraction(DualTensorOperation):
    name = 'Tensor Subtraction'

    def execute(self):
        return np.add(self._a, self._b)

    @staticmethod
    def vector_jacobian_product(g):
        return -1 * g, -1 * g


class TensorMultiply(DualTensorOperation): 
    name = 'Tensor Multiplication'

    def execute(self):
        return np.dot(self._a, self._b)

    def vector_jacobian_product(self):
        """ Returns the VJPs for the Jacobians of the arguments. """

        def a_vjp(self):
            """ g dotted with Jacobian of operation with respect to A"""
            a_sd, b_sd = self.is_single_dim()
            if b_sd:
                return lambda g: np.tensordot(g, self.b, 0)
            else:
                return lambda g: np.dot(g, np.swapaxes(self.b, -1, -2))

        def B_vjp(self):
            """ g dotted with Jacobian of operation with respect to B """
            a_sd, b_sd = self.is_single_dim()
            if a_sd:
                return lambda g: np.tensordot(g, self.a, 0)
            else:
                return lambda g: np.dot(g, np.swapaxes(self.a, -1, -2))

        return a_vjp(self), a_vjp(self)
