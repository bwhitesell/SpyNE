import numpy as np

from spyne.models import NeuralNetwork
from spyne.layers import FullyConnectedLayer


# generate some data
x = np.random.normal(0, 1, (50000, 5))
y = 3.14 * x[:, 0] + 1.41 * np.square(x[:, 1]) - 2.72 * np.power(x[:, 1], 4) - 1.62 * x[:, 4]

train_x = x[:40000]
train_y = y[:40000]

test_x = x[40000:]
test_y = y[40000:]


# build our nn
nn = NeuralNetwork()
nn.add_layer(FullyConnectedLayer(neurons=1000, activation='relu', dropout=.5))
nn.add_layer(FullyConnectedLayer(neurons=500, activation='relu', dropout=.5))
nn.add_layer(FullyConnectedLayer(neurons=20, activation='relu'))
nn.add_layer(FullyConnectedLayer(neurons=10, activation='relu'))
nn.add_layer(FullyConnectedLayer(neurons=1, activation='linear'))

print('\n')
print('Training a Multi-Layer-Perceptron...')
# fit our nn
nn.fit(train_x, train_y, batch_size=100, epochs=4, learning_rate=.001, l2=.05,
       loss='mse', optimizer='rmsprop', early_stopping=True)

