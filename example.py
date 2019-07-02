import numpy as np

from operations.arithmetic import TensorMultiply, TensorAddition
from operations.activations import TensorSigmoid
from variables.variables import Tensor, TensorConst
from differentiation.derivatives import ComputationGraph


def compute(x):
    a = TensorConst(x, name='x')

    # fully connected layer 1
    w1 = Tensor(np.random.random((50, 50)), name='w1')
    b1 = Tensor(np.random.random((50,)), name='b1')
    v1 = TensorMultiply(a, w1)
    a1 = TensorAddition(v1, b1)
    s1 = TensorSigmoid(a1)

    # fully connected layer 2
    w2 = Tensor(np.random.random((50,)), name='w2')
    e2 = TensorAddition(w2, b1)
    v2 = TensorMultiply(s1, e2)
    a2 = TensorAddition(v2, b1)
    s2 = TensorSigmoid(a2)

    return s2, ComputationGraph(s2).jacobians['b1']

print(
    compute(np.random.random((50,)))
)
