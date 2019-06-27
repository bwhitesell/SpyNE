import numpy as np

from .utils import is_ndarray, is_scalar



class OperationBase:
    name = 'Base Operation'

    def __init__(self, A, B):
        self.A = A
        self.B = B
        self._check_types()
        self.value = self.execute()

    def is_single_dim(self):
        return len(self.A.shape) == 1, len(self.B.shape) == 1

    def _check_types(self):
        if is_ndarray(self.A) and is_ndarray(self.B):
            if not is_scalar(self.A) or not is_scalar(self.B):
                return True

        raise ValueError(f"""Invalid types, {self.name} is an operation 
        between two tensors of at least dimension 1""")


class TensorAddition(OperationBase):
    name = 'Tensor Addition'

    def execute(self):
        np.add(self.A, self.B)

    def vector_jacobian_product(self):
        pass


class TensorMultiply(OperationBase):
    name = 'Tensor Multiplication'

    def execute(self):
        return np.dot(self.A, self.B)

    def vector_jacobian_product(self):
        """ Returns the VJPs for the Jacobians of the arguments. """

        def A_vjp(self):
            """ Jacobian of operation with respect to A """
            A_sd, B_sd = self.is_single_dim()
            if B_sd:
                return lambda G: np.tensordot(G, self.B, 0)
            else:
                return lambda G: np.dot(G, np.swapaxes(self.B, -1, -2))

        def B_vjp(self):
            """ Jacobian of operation with respect to B """
            A_sd, B_sd = self.is_single_dim()
            if A_sd:
                return lambda G: np.tensordot(G, self.A, 0)
            else:
                return lambda G: np.dot(G, np.swapaxes(self.A, -1, -2))

        return A_vjp(self), B_vjp(self)
