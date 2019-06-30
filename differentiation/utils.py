import numpy as np


def basis_vectors(x):
    for idxs in np.ndindex(*x.shape):
        vect = np.zeros(x.shape)
        vect[idxs] = 1
        yield vect