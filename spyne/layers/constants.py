import numpy as np
from spyne.operations.activations import TensorReLU, TensorSigmoid, TensorTanh


ACTIVATIONS = {
    'relu': TensorReLU,
    'sigmoid': TensorSigmoid,
    'tanh': TensorTanh,
    'linear': lambda x: x
}

XAVIER_INIT_PARAM = {
    'sigmoid': 1,
    'relu': np.sqrt(2),
    'tanh': 4,
    'linear': np.sqrt(2),
}


