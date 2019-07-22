import numpy as np

from spyne import Tensor, Constant


def basis_vectors(x):
    for idxs in np.ndindex(*x.shape):
        vect = np.zeros(x.shape)
        vect[idxs] = 1
        yield vect


def _is_tensor(node):
    return issubclass(node.__class__, Tensor) or issubclass(node.__class__, Constant)