import numpy as np

from spyne.models import NeuralNetwork
from spyne.layers import FullyConnectedLayer
from spyne.autodiff.operations.utils import one_hot_encode_categorical_target
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


# set seed

# load and standardize the data
wine = load_breast_cancer()
scaler = StandardScaler(with_mean=False)
x = scaler.fit_transform(wine['data'])
y = wine['target']
# build our nn
nn = NeuralNetwork()
nn.add_layer(FullyConnectedLayer(neurons=1, activation='sigmoid'))

print('\n')
print('Training a Multi-Layer-Perceptron...')
# fit our nn
nn.fit(x, y, batch_size=1, epochs=5, learning_rate=.00005, optimizer='sgd', loss='logloss', l2=0,
       early_stopping=False)

print(nn.predict([x[0]]))
print(y[0])





