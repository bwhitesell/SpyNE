import numpy as np

from variables.variables import Tensor, TensorConst


def basis_vectors(x):
    for idxs in np.ndindex(*x.shape):
        vect = np.zeros(x.shape)
        vect[idxs] = 1
        yield vect


def _is_tensor(node):
    return issubclass(node.__class__, Tensor) or issubclass(node.__class__, TensorConst)