from spyne.autodiff.operations.activations import TensorReLU, TensorSigmoid, TensorSoftmax


ACTIVATIONS = {
    'relu': TensorReLU,
    'sigmoid': TensorSigmoid,
    'softmax': TensorSoftmax,
    'tanh': TensorSigmoid,
    'linear': lambda x: x
}


