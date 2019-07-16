import numpy as np

from spyne.models import NeuralNetwork
from spyne.layers import FullyConnectedLayer

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler


# load and standardize the data
s = StandardScaler()
x = s.fit_transform(load_boston()['data'])
y = load_boston()['target']

# build our nn
nn = NeuralNetwork()
nn.add_layer(FullyConnectedLayer(neurons=100, activation='relu', dropout=.05))
nn.add_layer(FullyConnectedLayer(neurons=1, activation='linear'))

print('\n')
print('Training a Multi-Layer-Perceptron...')
# fit our nn
nn.fit(x, y, batch_size=1, epochs=10, learning_rate=.005, l2=.000001,
       loss='mse', optimizer='rmsprop', early_stopping=True)

