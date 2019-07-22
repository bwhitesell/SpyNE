import numpy as np

from .base import UniTensorOperation
from .utils import nest_func


class TensorSum(UniTensorOperation):
    name = 'Tensor Sum'

    def execute(self):
        return self._a.sum()

    def vector_jacobian_product(self, func=lambda g: g):
        internal_shape = self._a.shape

        @nest_func(func)
        def a_vjp(g):
            if g.shape == () or g.shape == (1,):
                return g * np.ones(internal_shape)
            else:
                raise NotImplementedError('''the VJP builder for TensorSum can only handle 
                                          single-axis arguments for now.''')
        return a_vjp


class TensorSquared(UniTensorOperation):
    name = 'Tensor Square'

    def execute(self):
        return np.square(self._a)

    def vector_jacobian_product(self, func=lambda g: g):
        a = self._a

        @nest_func(func)
        def a_vjp(g):
            return g * 2 * a

        return a_vjp


class TensorNegLog(UniTensorOperation):
    name = 'Tensor Element-Wise Negative Logarithm'

    def execute(self):
        return -1 * np.log(self._a)

    def vector_jacobian_product(self, func=lambda g: g):
        a = self._a

        @nest_func(func)
        def a_vjp(g):
            return np.divide(-g, a)

        return a_vjp


class TensorDuplicateRows(UniTensorOperation):
    name = 'Tensor Duplicate Rows'

    def __init__(self, a, n_rows):
        self.n_rows = n_rows
        super().__init__(a)
        if len(self._a.shape) != 1:
            raise ValueError(f'{self.name} only supports vectors for now.')

    def execute(self):
        return np.ones((self.n_rows,) + self._a.shape) * self._a

    def vector_jacobian_product(self, func=lambda g: g):
        @nest_func(func)
        def a_vjp(g):
            return np.sum(g, axis=0)

        return a_vjp