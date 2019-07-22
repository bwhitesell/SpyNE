import numpy as np
import unittest
from spyne import Tensor, Constant
from spyne.operations import (TensorAddition, TensorSubtraction, TensorMultiply, TensorElemMultiply, TensorReLU,
                              TensorTanh, TensorSigmoid, TensorSoftmax, TensorSum, TensorSquared, TensorNegLog,
                              TensorDuplicateRows)
from spyne.operations.base import DualTensorOperation, UniTensorOperation


class TestDualTensorBase(unittest.TestCase):
    operation = DualTensorOperation

    def setUp(self):
        self.A = Tensor(np.random.random((2, 2, 3)))
        self.B = Tensor(np.random.random((2, 2, 3)))

        self.C = self.operation(self.A, self.B)

    def test_constant_tensor_interchange(self):
        const_A = Constant(self.A.value)
        const_B = Constant(self.B.value)
        self.assertTrue(np.array_equal(self.C.value, self.operation(const_A, self.B).value))
        self.assertTrue(np.array_equal(self.C.value, self.operation(self.A, const_B).value))

    def test_attrs(self):
        self.assertEqual(self.C.a, self.A)
        self.assertEqual(self.C.b, self.B)

        self.assertTrue(np.array_equal(self.C._a, self.A.value))
        self.assertTrue(np.array_equal(self.C._b, self.B.value))

        self.assertTrue(type(self.C.node_uid) == str)
        self.assertTrue(len(self.C.node_uid) > 1)


class TestUniTensorBase(unittest.TestCase):
    operation = UniTensorOperation

    def setUp(self):
        self.A = Tensor(np.random.random((3, 5, 3)))
        self.B = self.operation(self.A)

    def test_constant_tensor_interchange(self):
        const_A = Constant(self.A.value)
        self.assertTrue(np.array_equal(self.B.value, self.operation(const_A).value))

    def test_attrs(self):
        self.assertEqual(self.B.a, self.A)

        self.assertTrue(np.array_equal(self.B._a, self.A.value))

        self.assertTrue(type(self.B.node_uid) == str)
        self.assertTrue(len(self.B.node_uid) > 1)


class TensorAdditionTest(TestDualTensorBase):
    operation = TensorAddition

    def test_execution(self):
        result = np.add(self.A.value, self.B.value)
        self.assertTrue(np.array_equal(self.C.value, result))
        self.assertEqual(self.C.shape, self.A.shape, self.B.shape)

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
        result = np.subtract(self.A.value, self.B.value)
        self.assertTrue(np.array_equal(self.C.value, result))
        self.assertEqual(self.C.shape, self.A.shape, self.B.shape)

    def test_vjps(self):
        g_1 = np.random.random((2,))
        g_2 = np.random.random((3, 2, 4))

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


class TensorMultiplyTest(TestDualTensorBase):
    operation = TensorMultiply

    def setUp(self):
        self.A = Tensor(
           np.random.random((2, 2, 3))
        )

        self.B = Tensor(
            np.random.random((2, 3, 2))
        )
        self.C = self.operation(self.A, self.B)

    def test_execution(self):
        self.assertTrue(np.array_equal(self.C.value, np.dot(self.A.value, self.B.value)))

    def test_vjps(self):
        # test case where a is of dim greater than 1 and b is of dim 1
        a = Tensor(np.random.random((3, 5, 4)))
        b = Tensor(np.random.random((4,)))
        c = self.operation(a, b)
        g = np.random.random(c.shape)
        contract_num = max(0, len(b.shape) - (len(a.shape) != 0))
        a_vjp, b_vjp = c.vector_jacobian_product()
        a_ndim = len(a.shape)
        b_ndim = len(b.shape)

        self.assertTrue(np.array_equal(a_vjp(g), np.tensordot(g, b.value, contract_num)))
        res = np.asarray(np.tensordot(
            g, a.value, [range(-a_ndim - b_ndim + 2, -b_ndim + 1), range(a_ndim - 1)]))
        self.assertTrue(np.array_equal(b_vjp(g), res))

        # test case where a is of dim 1 and b is of dim greater than 1
        a = Tensor(np.random.random((4,)))
        b = Tensor(np.random.random((3, 4, 5)))
        c = self.operation(a, b)
        g = np.random.random(c.shape)
        contract_num =  max(0, len(a.shape) - (len(b.shape) != 0))
        a_vjp, b_vjp = c.vector_jacobian_product()
        a_ndim = len(a.shape)
        b_ndim = len(b.shape)

        self.assertTrue(np.array_equal(a_vjp(g), np.tensordot(g, np.swapaxes(b.value, -1, -2), b_ndim - 1)))
        self.assertTrue(np.array_equal(b_vjp(g),
                                       np.asarray(np.swapaxes(np.tensordot(g, a.value, contract_num), -1, -2))))


class TestTensorElemMultiply(TestDualTensorBase):
    operation = TensorElemMultiply

    def test_execution(self):
        self.assertTrue(np.array_equal(self.C.value, np.multiply(self.A.value, self.B.value)))
        self.assertEqual(self.C.shape, self.A.shape, self.B.shape)

    def test_vjps(self):
        a_vjp, b_vjp = self.C.vector_jacobian_product()
        g = np.random.random((2, 2, 3))
        self.assertTrue(np.array_equal(a_vjp(g), np.multiply(g, self.B.value)))
        self.assertTrue(np.array_equal(b_vjp(g), np.multiply(g, self.A.value)))


class TestTensorRelu(TestUniTensorBase):
    operation = TensorReLU

    def test_execution(self):
        self.assertTrue(np.array_equal(self.B.value,  self.A.value * (self.A.value > 0)))
        self.assertEqual(self.B.shape, self.A.shape)

    def test_vjps(self):
        a_vjp = self.B.vector_jacobian_product()
        g = np.random.random(self.B.shape)
        self.assertTrue(np.array_equal(a_vjp(g), g * np.where(self.A.value > 0, 1, 0)))


class TestTensorTanh(TestUniTensorBase):
    operation = TensorTanh

    def test_execution(self):
        self.assertTrue(np.array_equal(self.B.value, np.tanh(self.A.value)))
        self.assertEqual(self.B.shape, self.A.shape)

    def test_vjps(self):
        a_vjp = self.B.vector_jacobian_product()
        g = np.random.random(self.B.shape)
        self.assertTrue(np.array_equal(a_vjp(g),  g * 1 / np.square(np.cosh(self.A.value))))


class TestTensorSigmoid(TestUniTensorBase):
    operation = TensorSigmoid

    def test_execution(self):
        e = np.exp(self.A.value)
        self.assertTrue(np.array_equal(self.B.value, e / (e + 1)))
        self.assertEqual(self.B.shape, self.A.shape)

    def test_vjps(self):
        a_vjp = self.B.vector_jacobian_product()
        g = np.random.random(self.B.shape)
        e = np.exp(-1 * self.A.value)
        self.assertTrue(np.array_equal(a_vjp(g),  np.multiply(g, e / (1 + e)**2)))


class TestTensorSoftmax(TestUniTensorBase):
    operation = TensorSoftmax

    def test_execution(self):
        exp_sum = np.exp(self.A.value).sum()
        self.assertTrue(np.array_equal(self.B.value, np.exp(self.A.value) / exp_sum))
        self.assertEqual(self.B.shape, self.A.shape)

    def test_vjps(self):
        a_vjp = self.B.vector_jacobian_product()
        g = np.random.random(self.B.shape)
        exp_sum = np.exp(self.A.value).sum()
        num = np.multiply(np.exp(self.A.value), exp_sum) - np.exp(2 * self.A.value)
        self.assertTrue(np.array_equal(a_vjp(g), np.multiply(g, num / np.square(exp_sum))))


class TestTensorSum(TestUniTensorBase):
    operation = TensorSum

    def test_execution(self):
        self.assertTrue(np.array_equal(self.B.value, self.A.value.sum()))
        self.assertEqual(self.B.shape, ())

    def test_vjps(self):
        a_vjp = self.B.vector_jacobian_product()
        g = np.random.random((1,))
        self.assertTrue(np.array_equal(a_vjp(g), g * np.ones(self.A.shape)))


class TestTensorSquared(TestUniTensorBase):
    operation = TensorSquared

    def test_execution(self):
        self.assertTrue(np.array_equal(self.B.value, np.square(self.A.value)))
        self.assertEqual(self.B.shape, self.A.shape)

    def test_vjps(self):
        a_vjp = self.B.vector_jacobian_product()
        g = np.random.random(self.A.shape)
        self.assertTrue(np.array_equal(a_vjp(g), g * 2 * self.A.value))


class TestTensorNegLog(TestUniTensorBase):
    operation = TensorNegLog

    def test_execution(self):
        self.assertTrue(np.array_equal(self.B.value, -1 * np.log(self.A.value)))
        self.assertEqual(self.B.shape, self.A.shape)

    def test_vjps(self):
        a_vjp = self.B.vector_jacobian_product()
        g = np.random.random(self.A.shape)
        self.assertTrue(np.array_equal(a_vjp(g), np.divide(-g, self.A.value)))


class TestTensorDuplicateRows(unittest.TestCase):
    operation = TensorDuplicateRows

    def setUp(self):
        self.A = Tensor(np.random.random((7,)))
        self.n_rows = 5
        self.B = self.operation(self.A, self.n_rows)

    def test_execution(self):
        self.assertTrue(np.array_equal(self.B.value, np.ones((self.n_rows,) + self.A.shape) * self.A.value))
        self.assertEqual(self.B.shape, (self.n_rows,) + self.A.shape)

    def test_vjps(self):
        a_vjp = self.B.vector_jacobian_product()
        g = np.random.random(self.A.shape)
        self.assertTrue(np.array_equal(a_vjp(g), np.sum(g, axis=0)))
