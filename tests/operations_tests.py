import numpy as np
import unittest
from spyne import Tensor, Constant
from spyne.operations.arithmetic import TensorAddition, TensorSubtraction
from spyne.operations.base import DualTensorOperation, BaseOperation


class TestDualTensorBase(unittest.TestCase):
    operation = DualTensorOperation

    def setUp(self):
        self.A = Tensor(
            [
                [[1, 2, 3],
                 [4, 5, 6]],

                [[7, 8, 9],
                 [10, 11, 12]],
            ]
        )

        self.B = Tensor(
            [
                [[2, 3, 2.2],
                 [1, 4, 8]],

                [[3, 5, 6],
                 [9, 4, 1.09]],
            ]
        )

        self.C = self.operation(self.A, self.B)

    def test_constant_tensor_interchange(self):
        const_A = Constant(self.A.value)
        const_B = Constant(self.B.value)
        self.assertTrue(np.array_equal(self.C.value, self.operation(const_A, self.B).value))
        self.assertTrue(np.array_equal(self.C.value, self.operation(self.A, const_B).value))

    def test_is_single_dim(self):
        a_single, b_single = self.C.is_single_dim()
        self.assertFalse(a_single)
        self.assertFalse(b_single)

        c = self.operation(self.A, Tensor([1]))
        a_single, b_single = c.is_single_dim()
        self.assertFalse(a_single)
        self.assertTrue(b_single)

    def test_attrs(self):
        self.assertEqual(self.C.a, self.A)
        self.assertEqual(self.C.b, self.B)

        self.assertTrue(np.array_equal(self.C._a, self.A.value))
        self.assertTrue(np.array_equal(self.C._b, self.B.value))

        self.assertTrue(type(self.C.node_uid) == str)
        self.assertTrue(len(self.C.node_uid) > 1)


class TensorAdditionTest(TestDualTensorBase):
    operation = TensorAddition

    def test_execution(self):
        result = np.array(
            [
                [[3, 5, 5.2],
                 [5, 9, 14]],

                [[10, 13, 15],
                 [19, 15, 13.09]],
            ]
        )
        self.assertTrue(np.array_equal(self.C.value, result))

    def test_vjps(self):
        g_1 = np.array([1, 2])
        g_2 = np.array([[[4, 5, 22, 33], [1, 2, 3, 4]], [[1, 2, 3, 4], [5, 6, 7, 8]]])

        a_vjp, b_vjp = self.C.vector_jacobian_product()

        self.assertTrue(np.array_equal(a_vjp(g_1), g_1))
        self.assertTrue(np.array_equal(a_vjp(g_2), g_2))
        self.assertTrue(np.array_equal(b_vjp(g_1), g_1))
        self.assertTrue(np.array_equal(b_vjp(g_2), g_2))

    def test_attrs(self):
        self.assertEqual(self.C.shape, (2, 2, 3))

    def test_methods(self):
        self.assertTrue(self.C.is_tensor(self.A))
        self.assertFalse(self.C.is_tensor(np.array(1)))

        self.assertFalse(self.C.is_operation(self.A))
        self.assertTrue(self.C.is_operation(self.C))


class TensorSubtractionTest(TestDualTensorBase):
    operation = TensorSubtraction

    def test_execution(self):
        print(self.C.value)
        result = np.array(
            [
                [[-1, -1, 0.8],
                 [3, 1, -2]],

                [[4, 3, 3],
                 [1, 7, 10.91]],
            ]
        )
        self.assertTrue(np.array_equal(self.C.value, result))

    def test_vjps(self):
        g_1 = np.array([1, 2])
        g_2 = np.array([[[4, 5, 22, 33], [1, 2, 3, 4]], [[1, 2, 3, 4], [5, 6, 7, 8]]])

        a_vjp, b_vjp = self.C.vector_jacobian_product()

        self.assertTrue(np.array_equal(a_vjp(g_1), -1 * g_1))
        self.assertTrue(np.array_equal(a_vjp(g_2), -1 * g_2))
        self.assertTrue(np.array_equal(b_vjp(g_1), -1 * g_1))
        self.assertTrue(np.array_equal(b_vjp(g_2), -1 * g_2))

    def test_attrs(self):
        self.assertEqual(self.C.shape, (2, 2, 3))

    def test_methods(self):
        self.assertTrue(self.C.is_tensor(self.A))
        self.assertFalse(self.C.is_tensor(np.array(1)))

        self.assertFalse(self.C.is_operation(self.A))
        self.assertTrue(self.C.is_operation(self.C))




