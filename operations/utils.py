import numpy as np


def is_ndarray(x):
    return type(x) == np.ndarray


def is_scalar(x):
    return x.shape == ()
