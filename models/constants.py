from autodiff.operations.activations import TensorReLU, TensorTanh, TensorSigmoid


ACTIVATIONS = {
    'relu': TensorReLU,
    'sigmoid': TensorSigmoid,
    'tanh': TensorSigmoid,
    'linear': lambda x: x
}


