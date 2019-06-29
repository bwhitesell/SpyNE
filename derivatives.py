import numpy as np

from operations.base import UniTensorOperation, DualTensorOperation


def jacobian(self, dep_var, ind_var):
    bases = basis_vectors(dep_var)

    if issubclass(dep_var, UniTensorOperation):
        g_a = dep_var.vector_jacobian_product()
    if issubclass(dep_var, DualTensorOperation):
        g_a, g_b = dep_var.vector_jacobian_product()





def basis_vectors(x):
    for idxs in np.ndindex(*x.shape):
        vect = np.zeros(x.shape)
        vect[idxs] = 1
        yield vect