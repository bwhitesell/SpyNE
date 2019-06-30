import numpy as np

from .base import UniTensorOperation


class TensorSum(UniTensorOperation):
    name = 'Tensor Sum'

    def execute(self):
        return self._a.sum()

    def vector_jacobian_product(self, g):
        internal_shape = self._a.shape
        if g.shape == ():
            return g * np.ones(internal_shape)
        else:
            raise NotImplementedError('''the VJP of TensorSum can only handle 
                                      scalar arguments for now.''')

class TensorSquared(UniTensorOperation):
    name = 'Tensor Square'

    def execute(self):
        return np.square(self._a)

    def vector_jacobian_product(self, g):
        return g * 2 * self._a