import numpy as np

from ..variables.variables import Tensor, TensorConst


def is_tensor(x):
    return type(x) == Tensor or type(x) == TensorConst


def nest_func(external_func):
    def func_construction(internal_func):
        def wrapped_func(*args, **kwargs):
            g = external_func(*args, **kwargs)
            return internal_func(g)

        return wrapped_func
    return func_construction


def one_hot_encode_categorical_target(y):
    n_categories = np.unique(y).size
    n_samples = y.size
    y_adj = np.zeros((n_samples, n_categories))
    y_adj[np.arange(n_samples), y-1] = 1
    return y_adj


