import numpy as np

from operations.arithmetic import TensorAddition
from operations.activations import TensorSigmoid
from variables.variables import Tensor, TensorConst
from differentiation.derivatives import Gradient
import time


def compute(x):
    a = Tensor(x, name='x')
    b = TensorConst(np.random.random((50,50)), name='b')
    c = TensorAddition(a, b)
    d = TensorSigmoid(c)
    e = TensorAddition(c, d)

    return e, Gradient(e).jacobians['x']

print(compute(np.random.random((50,50))))

