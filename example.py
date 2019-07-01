import numpy as np

from operations.arithmetic import TensorMultiply, TensorAddition
from operations.activations import TensorSigmoid
from variables.variables import Tensor, TensorConst
from differentiation.derivatives import Gradient


def compute(x):
    a = TensorConst(x, name='x')

    # fully connected layer 1
    w1 = Tensor(np.random.random((50, 50)), name='w1')
    b1 = Tensor(np.random.random((50,)), name='b1')
    v1 = TensorMultiply(a, w1)
    s1 = TensorAddition(v1, b1)
    a1 = TensorSigmoid(s1)

    # fully connected layer 2
    w2 = Tensor(np.random.random((50,)), name='w2')
    b2 = Tensor(np.random.random((1,)), name='b2')
    v2 = TensorMultiply(a1, w2)
    s2 = TensorAddition(v2, b2)
    a2 = TensorSigmoid(s2)



    return a2, Gradient(a2).jacobians

print(
    compute(np.random.random((50,)))
)

