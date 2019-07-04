import numpy as np

from models.graphs import NeuralNetwork, FullyConnectedLayer
from autodiff.variables.variables import Tensor, TensorConst


nn = NeuralNetwork()
nn.add_layer(FullyConnectedLayer(neurons=50, activation='relu'))
nn.add_layer(FullyConnectedLayer(neurons=25, activation='sigmoid'))
nn.add_layer(FullyConnectedLayer(neurons=10, activation='relu'))
nn.add_layer(FullyConnectedLayer(neurons=5, activation='relu'))
nn.add_layer(FullyConnectedLayer(neurons=1, activation='linear'))




x = TensorConst([.1, .5, 3, 2, 2, 5, 4])
y = TensorConst([500])

nn._setup_layers(x)

print('Learning relationship...')
for i in range(100):
    print('----- Pass -----')
    nn._learn_iter(x, y)


