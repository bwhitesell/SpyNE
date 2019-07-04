import numpy as np

from models.graphs import NeuralNetwork, FullyConnectedLayer
from autodiff.variables.variables import Tensor, TensorConst

x = Tensor([.1, .5, 3, 2, 2, 5, 4])

nn = NeuralNetwork()
nn.add_layer(FullyConnectedLayer(neurons=5, activation='linear'))
nn.add_layer(FullyConnectedLayer(neurons=1, activation='linear'))


y = TensorConst([1250])

print('Learning relationship...')
for i in range(100):
    print('----- Pass -----')
    nn._learn_row(x, y)


