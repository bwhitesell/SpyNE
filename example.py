import numpy as np

from spyne.models import NeuralNetwork
from spyne.layers import FullyConnectedLayer

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler


# load and standardize the data
s = StandardScaler()
x = s.fit_transform(load_boston()['data'])
y = load_boston()['target']

#build our nn
nn = NeuralNetwork()
nn.add_layer(FullyConnectedLayer(neurons=5, activation='relu'))
nn.add_layer(FullyConnectedLayer(neurons=5, activation='relu'))
nn.add_layer(FullyConnectedLayer(neurons=1, activation='linear'))

print('\n')
print('Training a Multi-Layer-Perceptron...')
# fit our nn
nn.fit(x, y, batch_size=1, epochs=100, learning_rate=.003, loss='mse', optimizer='rmsprop', l2=.01, early_stopping=False)

