import numpy as np

from operations.arithmetic import TensorMultiply, TensorAddition
from operations.activations import TensorReLU
from variables.variables import Tensor, TensorConst
from autodiff.derivatives import BackwardsPass


def neural_network(x):
    """
    An example of the models that can be built with SpyNE's Automatic
    Differentiation. Specifically, this is a fully connected artificial
    neural network with one hidden layer. S2 is calculated through the
    forward pass, then the BackwardsPass class executes the backwards
    pass. Notice we get the jacobian of all variables returned which can
    be used to optimize the model parameters w1, b1, w2, b2.
    """

    a = TensorConst(x, name='x')

    # fully connected layer 1
    w1 = Tensor(np.random.random((50, 50)), name='w1')
    b1 = Tensor(np.random.random((50,)), name='b1')
    v1 = TensorMultiply(a, w1)
    a1 = TensorAddition(v1, b1)
    s1 = TensorReLU(a1)

    # fully connected layer 2
    w2 = Tensor(np.random.random((50,)), name='w2')
    b2 = Tensor(np.random.random((1,)), name='b2')
    v2 = TensorMultiply(s1, w2)
    s2 = TensorReLU(v2)

    return s2, BackwardsPass(s2).jacobians['w1']


print(
    neural_network(np.random.random((50,)))
)
