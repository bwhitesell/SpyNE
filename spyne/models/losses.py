from spyne.autodiff.operations.arithmetic import TensorSubtraction, TensorElemMultiply
from spyne.autodiff.operations.elements import TensorSquared, TensorSum, TensorNegLog


def mean_squared_error(y, y_hat):
    error = TensorSubtraction(y, y_hat)
    error_sq = TensorSquared(error)
    return TensorSum(error_sq)


def log_loss(y, y_hat):
    yh_neg_log = TensorNegLog(y_hat)
    lp = TensorElemMultiply(yh_neg_log, y)
    return TensorSum(lp)


LOSSES = {
    'mse': mean_squared_error,
    'logloss': log_loss,
}
