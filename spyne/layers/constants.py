from spyne.autodiff.operations.activations import TensorReLU, TensorSigmoid, TensorSoftmax, TensorTanh


ACTIVATIONS = {
    'relu': TensorReLU,
    'sigmoid': TensorSigmoid,
    'softmax': TensorSoftmax,
    'tanh': TensorTanh,
    'linear': lambda x: x
}


