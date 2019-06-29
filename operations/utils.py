import numpy as np

from variables import Tensor, TensorConst


def is_tensor(x):
    return type(x) == Tensor or type(x) == TensorConst

