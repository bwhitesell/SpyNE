import numpy as np

from operations.arithmetic import TensorAddition
from operations.activations import TensorSigmoid
from variables.variables import Tensor, TensorConst
from differentiation.derivatives import Gradient
import time


def compute(x):
    a = Tensor(x, name='x')
    b = TensorConst([[1, 2], [3, 4]], name='b')
    c = TensorAddition(a, b)
    d = TensorSigmoid(c)
    e = TensorAddition(c, d)

    return e, Gradient(e).jacobians['x']

print(
    compute([[1, 2], [3, 4]])
)

