import numpy as np

from spyne.autodiff.operations.arithmetic import TensorSubtraction, TensorElemMultiply
from spyne.autodiff.operations.elements import TensorSquared, TensorSum, TensorNegLog
from spyne.autodiff.variables.variables import TensorConst


def mean_squared_error(y, y_hat):
    error = TensorSubtraction(y, y_hat)
    error_sq = TensorSquared(error)
    return TensorSum(error_sq)


def log_loss(y, y_hat):
    """ The log loss of a single prediction and target outcome. """
    rect_pos = TensorElemMultiply(y_hat, y)

    prob_neg = TensorSubtraction(TensorConst(np.ones(y_hat.shape)), y_hat)
    neg_outcome = TensorSubtraction(TensorConst(np.ones(y.shape)), y)
    rect_neg = TensorElemMultiply(prob_neg, neg_outcome)



    lp = TensorElemMultiply(yh_neg_log, y)
    print(lp)
    return TensorSum(lp)


LOSSES = {
    'mse': mean_squared_error,
    'logloss': log_loss,
}
