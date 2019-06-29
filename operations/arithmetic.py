import numpy as np

from .base import UniTensorOperation, DualTensorOperation
from variables import Tensor

class TensorAddition(DualTensorOperation):
    name = 'Tensor Addition'

    def execute(self):
        return Tensor(np.add(self.a, self.b))

    @staticmethod
    def vector_jacobian_product(self):
        def a_vjp(g):
            return g

        def b_vjp(g):
            return g

        return a_vjp, b_vjp


class TensorSubtraction(DualTensorOperation):
    name = 'Tensor Subtraction'

    def execute(self):
        return Tensor(np.add(self.a, self.b))

    @staticmethod
    def vector_jacobian_product(self):
        def a_vjp(g):
            return -1 * g

        def b_vjp(g):
            return -1 * g

        return a_vjp, b_vjp


class TensorMultiply(DualTensorOperation): 
    name = 'Tensor Multiplication'

    def execute(self):
        return Tensor(np.dot(self.a, self.b))

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
