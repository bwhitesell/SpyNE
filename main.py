import numpy as np

from .utils import is_ndarray, is_scalar


class OperationBase:
    name = 'Base Operation'

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self._check_types()
        self.value = self.execute()

    def execute(self):
        pass

    def is_single_dim(self):
        return len(self.a.shape) == 1, len(self.a.shape) == 1

    def _check_types(self):
        if is_ndarray(self.a) and is_ndarray(self.b):
            if not is_scalar(self.a) or not is_scalar(self.b):
                return True

        raise ValueError(f"""Invalid types, {self.name} is an operation 
        between two tensors of at least dimension 1""")


class TensorAddition(OperationBase):
    name = 'Tensor Addition'

    def execute(self):
        np.add(self.a, self.a)

    @staticmethod
    def vector_jacobian_product(self):
        def a_vjp(g):
            return g

        def b_vjp(g):
            return g

        return a_vjp, b_vjp


class TensorSubtraction(OperationBase):
    name = 'Tensor Subtraction'

    def execute(self):
        np.add(self.a, self.b)

    @staticmethod
    def vector_jacobian_product(self):
        def a_vjp(g):
            return -1 * g

        def b_vjp(g):
            return -1 * g

        return a_vjp, b_vjp


class TensorReLU(OperationBase):
    name = 'Tensor ReLU'

    def __init__(self, a):
        self.a = a
        self._check_types()
        self.value = self.execute()

    def execute(self):
        return self.a * (self.a > 0)

    def vector_jacobian_product(self):
        def a_vjp(g):
           return g * np.where(self.a > 0, 1, 0)
        return a_vjp


class TensorTanh(OperationBase):
    name = 'Tensor Tanh'

    def __init__(self, a):
        self.a = a
        self._check_types()
        self.value = self.execute()

    def execute(self):
        return np.tanh(self.a)

    def vector_jacobian_product(self):
        def a_vjp(g):
            return g * 1 / np.square(np.cosh(self.a))
        return a_vjp


class TensorSigmoid(OperationBase):
    name = 'Tensor Sigmoid'

    def __init__(self, a):
        self.a = a
        self._check_types()
        self.value = self.execute()

    def execute(self):
        e = np.exp(self.a)
        return e / (e + 1)

    def vector_jacobian_product(self):
        def a_vjp(g):
            e = np.exp(self.a)
            return g * (e / (e**2 + 1))

        return a_vjp


class Tensor

class TensorMultiply(OperationBase):
    name = 'Tensor Multiplication'

    def execute(self):
        return np.dot(self.a, self.b)

    def vector_jacobian_product(self):
        """ Returns the VJPs for the Jacobians of the arguments. """

        def a_vjp(self):
            """ Jacobian of operation with respect to A """
            a_sd, b_sd = self.is_single_dim()
            if b_sd:
                return lambda g: np.tensordot(g, self.b, 0)
            else:
                return lambda g: np.dot(g, np.swapaxes(self.b, -1, -2))

        def B_vjp(self):
            """ Jacobian of operation with respect to B """
            a_sd, b_sd = self.is_single_dim()
            if a_sd:
                return lambda g: np.tensordot(g, self.a, 0)
            else:
                return lambda g: np.dot(g, np.swapaxes(self.a, -1, -2))

        return a_vjp(self), a_vjp(self)
