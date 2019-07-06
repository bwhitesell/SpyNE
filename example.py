import numpy as np

from models.graphs import NeuralNetwork
from layers import FullyConnectedLayer


x = np.random.random((5000, 224))
y = x.sum(axis=1)


nn = NeuralNetwork()
nn.add_layer(FullyConnectedLayer(neurons=50, activation='relu'))
nn.add_layer(FullyConnectedLayer(neurons=1, activation='linear'))


print('Learning relationship...')

nn.fit(x, y, batch_size=1000, epochs=5, learning_rate=.5, optimizer='adam')


