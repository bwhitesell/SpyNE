import numpy as np
import unittest
from spyne.layers import FullyConnectedLayer


class TestFullyConnectedLayer(unittest.TestCase):

    def setUp(self):
        self.x = np.random.random((254, 15))

    def test_setup(self):
        neurons = 12
        l1 = FullyConnectedLayer(neurons=neurons)
        l1.setup(self.x)

        self.assertEqual(l1.w.shape, (15, neurons))
        self.assertEqual(l1._b.shape, (neurons,))
        self.assertEqual(l1.b.shape, (254, neurons))
