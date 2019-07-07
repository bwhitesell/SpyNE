from spyne.autodiff.operations.arithmetic import TensorSubtraction
from spyne.autodiff.operations.elements import TensorSquared, TensorSum


def mean_squared_error(y, y_hat):
    error = TensorSubtraction(y, y_hat)
    error_sq = TensorSquared(error)
    return TensorSum(error_sq)


LOSSES = {
    'mse': mean_squared_error,
}