import numpy as np
from spyne import Tensor, Constant
from spyne.operations import TensorSubtraction, TensorElemMultiply, TensorAddition
from spyne.operations import TensorSquared, TensorSum, TensorNegLog


def mean_squared_error(y, y_hat):
    error = TensorSubtraction(y, y_hat)
    error_sq = TensorSquared(error)
    return TensorSum(error_sq)


def log_loss(y, y_hat):
    """ The log loss of a single prediction and target outcome. """
    pos_prob_log = TensorNegLog(y_hat)
    pos_prob_rect = TensorElemMultiply(pos_prob_log, y)

    neg_prob = TensorSubtraction(
        Constant(np.ones(y_hat.shape)),
        y_hat,
    )
    neg_prob_log = TensorNegLog(neg_prob)
    neg_outcomes = TensorSubtraction(
        Constant(np.ones(y.shape)),
        y,
    )
    neg_prob_rect = TensorElemMultiply(neg_prob_log, neg_outcomes)

    log_probs = TensorAddition(neg_prob_rect, pos_prob_rect)

    return TensorSum(log_probs)


LOSSES = {
    'mse': mean_squared_error,
    'logloss': log_loss,
}
