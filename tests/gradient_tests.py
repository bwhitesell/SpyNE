import unittest
import numpy as np
from spyne import Tensor, Constant
from spyne.gradients import BackwardsPass
from spyne.operations import TensorMultiply, TensorSubtraction, TensorAddition, TensorNegLog, TensorElemMultiply, TensorSum


class TestBackwardsPass(unittest.TestCase):

    def setUp(self):
        self.A = Tensor([[1, 2], [3, 4]], name='A')
        self.B = Tensor([[1, 2], [3, 4]], name='B')
        self.C = Tensor([[5, 6], [7, 8]], name='C')
        self.D = Constant([[2, 1], [3, 2]], name='D')

        # Define an involved computation graph with the constants and variables above
        e = TensorAddition(self.A, self.B)
        f = TensorElemMultiply(Constant(2 * np.ones(e.shape)), e)
        g = TensorNegLog(f)
        h = TensorAddition(self.A, g)
        i = TensorMultiply(h, self.C)
        j = TensorSubtraction(i, self.D)
        l = TensorSum(j)

        self.bp = BackwardsPass(l)

    def test_constant_variable_parsing(self):
        self.assertTrue('A' in self.bp.vjps)
        self.assertTrue('B' in self.bp.vjps)
        self.assertTrue('C' in self.bp.vjps)
        self.assertFalse('D' in self.bp.vjps)

    def test_jacobian_construction(self):
        self.assertTrue(np.array_equal(
            self.bp.execute()['A'], np.array([[-5.5, -11.25], [-9.166666666666666, -13.125]])
        ))
        self.assertTrue(np.array_equal(
            self.bp.execute()['B'], np.array([[5.5, 3.75], [1.8333333333333333, 1.875]])
        ))
        self.assertTrue(np.array_equal(
            self.bp.execute()['C'], np.array([[-0.12879898909210907, -0.12879898909210907], [-1.147969736080383, -1.147969736080383]])
        ))



