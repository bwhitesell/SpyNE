import unittest
import numpy as np
from spyne import Tensor, Constant


class TestTensor(unittest.TestCase):
    """ A set of tests to validate tensor definitions, attributes, functionality etc..."""

    def setUp(self):
        self.data = np.array(
            [
                [[1, 2, 3],
                 [4, 5, 6]],

                [[7, 8, 9],
                 [10, 11, 12]],
            ]
        )

        self.tens = Tensor(self.data)

    def test_tensor_attributes(self):
        """ Test tensor attributes """

        # value attr should return the wrapped data
        self.assertTrue(np.array_equal(self.tens.value, self.data))
        # shape should return the shape of the wrapped data
        self.assertEqual(self.tens.shape, (2, 2, 3))
        # node_uid should be a random string generated uniquely for each tensor
        self.assertTrue(type(self.tens.node_uid) == str)
        self.assertTrue(len(self.tens.node_uid) > 0)
        new_tens = Tensor(self.tens.value)
        self.assertNotEqual(self.tens.node_uid, new_tens.node_uid)

    def test_tensor_instantiation(self):
        """ Test tensor instantiation approaches """

        new_tensor = Tensor(self.data.tolist())
        # tensors can accept lists on instantiation and will internally convert the data to an ndarray
        self.assertTrue(np.array_equal(new_tensor.value, self.tens.value))


# for now tensors and constants share identical functionality. In the future they may not.
class TestConstant(unittest.TestCase):
    """ A set of tests to validate Constant's definitions, attributes, functionality etc..."""

    def setUp(self):
        self.data = np.array(
            [
                [[1, 2, 3],
                 [4, 5, 6]],

                [[7, 8, 9],
                 [10, 11, 12]],
            ]
        )

        self.const = Constant(self.data)

    def test_constants_attributes(self):
        """ Test tensor attributes """

        # value attr should return the wrapped data
        self.assertTrue(np.array_equal(self.const.value, self.data))
        # shape should return the shape of the wrapped data
        self.assertEqual(self.const.shape, (2, 2, 3))
        # node_uid should be a random string generated uniquely for each tensor
        self.assertTrue(type(self.const.node_uid) == str)
        self.assertTrue(len(self.const.node_uid) > 0)
        new_const = Tensor(self.const.value)
        self.assertNotEqual(self.const.node_uid, new_const.node_uid)

    def test_constants_instantiation(self):
        """ Test tensor instantiation approaches """

        new_const = Tensor(self.data.tolist())
        # tensors can accept lists on instantiation and will internally convert the data to an ndarray
        self.assertTrue(np.array_equal(new_const.value, self.const.value))


if __name__ == '__main__':
    unittest.main()