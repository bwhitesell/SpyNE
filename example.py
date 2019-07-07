import numpy as np

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
boston = load_boston()

from models.graphs import NeuralNetwork
from layers import FullyConnectedLayer


# set seed
np.random.seed(1)

# standardize the data
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


# linear regression
print('\n')
print('Or a linear regression....')
linear_model = NeuralNetwork()
linear_model.add_layer(FullyConnectedLayer(neurons=1, activation='linear', dropout=0))
linear_model.fit(x, y, batch_size=10, epochs=20, learning_rate=.005, optimizer='rmsprop', l2=0.0000001)





