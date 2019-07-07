import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
boston = load_boston()

from models.graphs import NeuralNetwork
from layers import FullyConnectedLayer


# standardize the data
scaler = StandardScaler(with_mean=False)
x = scaler.fit_transform(boston['data'])
y = boston['target']

# build our nn
nn = NeuralNetwork()
nn.add_layer(FullyConnectedLayer(neurons=5, activation='relu', dropout=0))
nn.add_layer(FullyConnectedLayer(neurons=1, activation='linear'))


# fit our nn
nn.fit(x, y, batch_size=10, epochs=5, learning_rate=.1, optimizer='rmsprop', l2=0.0000001)


