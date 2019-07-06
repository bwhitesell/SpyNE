from autodiff.operations.arithmetic import TensorSubtraction, TensorAddition
from autodiff.operations.elements import TensorSquared, TensorSum


def mean_squared_error(y, y_hat):
    error = TensorSubtraction(y, y_hat)
    error_sq = TensorSquared(error)
    return TensorSum(error_sq)


LOSSES = {
    'mse': mean_squared_error,
}