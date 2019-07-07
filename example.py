import numpy as np

from spyne.models import NeuralNetwork
from spyne.layers import FullyConnectedLayer

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler


# set seed
np.random.seed(1)

# load and standardize the data
boston = load_boston()
scaler = StandardScaler(with_mean=False)
x = scaler.fit_transform(boston['data'])
y = boston['target']

# build our nn
nn = NeuralNetwork()
nn.add_layer(FullyConnectedLayer(neurons=5, activation='relu', dropout=0))
nn.add_layer(FullyConnectedLayer(neurons=1, activation='linear'))

print('\n')
print('Training a Multi-Layer-Perceptron...')
# fit our nn
nn.fit(x, y, batch_size=10, epochs=20, learning_rate=.09, optimizer='rmsprop', l2=0.0000001)





