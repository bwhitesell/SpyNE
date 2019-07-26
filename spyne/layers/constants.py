from spyne.operations.activations import TensorReLU, TensorSigmoid, TensorTanh


ACTIVATIONS = {
    'relu': TensorReLU,
    'sigmoid': TensorSigmoid,
    'tanh': TensorTanh,
    'linear': lambda x: x
}


