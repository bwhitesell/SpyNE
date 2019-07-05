import numpy as np

from models.graphs import NeuralNetwork, FullyConnectedLayer
from autodiff.variables.variables import Tensor, TensorConst

x = np.array([[.1, .5, 3, 2, 2, 5, 4],
              [.1, .5, 3, 2, 2, 5, 4],
              [.1, .5, 3, 2, 2, 5, 4],
              [.1, .5, 3, 2, 2, 5, 4]])
y = np.array([[500],
              [500],
              [500],
              [500]])


nn = NeuralNetwork()
nn.add_layer(FullyConnectedLayer(neurons=5000, activation='relu'))
nn.add_layer(FullyConnectedLayer(neurons=2500, activation='relu'))
nn.add_layer(FullyConnectedLayer(neurons=1000, activation='relu'))
nn.add_layer(FullyConnectedLayer(neurons=500, activation='relu'))
nn.add_layer(FullyConnectedLayer(neurons=1, activation='linear'))


print('Learning relationship...')

nn.fit(x, y, batch_size=4, epochs=10)


