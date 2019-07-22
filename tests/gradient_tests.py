import unittest
import numpy as np
from spyne import Tensor, Constant
from spyne.gradients import BackwardsPass
from spyne.operations import TensorMultiply, TensorSubtraction, TensorAddition, TensorNegLog, TensorReLU, TensorElemMultiply


class TestBackwardsPass(unittest.TestCase):

    def setUp(self):
        self.A = Tensor(np.random.random((6, 4, 2, 3)), name='A')
        self.B = Tensor(np.random.random((6, 4, 2, 3)), name='B')
        self.C = Tensor(np.random.random((15, 22, 3, 4)), name='C')
        self.D = Constant(np.random.random((6, 4, 2, 15, 22, 4)), name='D')

        # Define an involved computation graph with the constants and variables above
        e = TensorAddition(self.A, self.B)
        f = TensorElemMultiply(Constant(2 * np.ones(self.e.shape)), e)
        g = TensorNegLog(f)
        h = TensorAddition(self.A, g)
        i = TensorMultiply(h, self.C)
        j = TensorSubtraction(i, self.D)

        self.bp = BackwardsPass(j)

    def test_constant_variable_parsing(self):
        self.assertTrue('A' in self.bp.vjps)

