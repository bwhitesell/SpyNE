import numpy as np

from spyne.models import NeuralNetwork
from spyne.layers import FullyConnectedLayer
from spyne.autodiff.operations.utils import one_hot_encode_categorical_target
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


# set seed

# load and standardize the data
wine = load_wine()
scaler = StandardScaler(with_mean=False)
x = scaler.fit_transform(wine['data'])
y = one_hot_encode_categorical_target(wine['target'])
# build our nn
nn = NeuralNetwork()
nn.add_layer(FullyConnectedLayer(neurons=5, activation='relu', dropout=0))
nn.add_layer(FullyConnectedLayer(neurons=3, activation='softmax'))

print('\n')
print('Training a Multi-Layer-Perceptron...')
# fit our nn
nn.fit(x, y, batch_size=1, epochs=10, learning_rate=.001, optimizer='rmsprop', loss='logloss', l2=0,
       early_stopping=False)

print(nn.predict([x[0]]))
print(y[0])





