from spyne.autodiff.operations.activations import TensorReLU, TensorSigmoid


ACTIVATIONS = {
    'relu': TensorReLU,
    'sigmoid': TensorSigmoid,
    'tanh': TensorSigmoid,
    'linear': lambda x: x
}


