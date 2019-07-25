from spyne.models import NeuralNetwork
from spyne.layers import FullyConnectedLayer
from spyne.operations.utils import one_hot_encode_categorical_target
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


# set seed

# load and standardize the data
wine = load_wine()
scaler = StandardScaler(with_mean=True)
x = scaler.fit_transform(wine['data'])
y = one_hot_encode_categorical_target(wine['target'])
# build our nn
nn = NeuralNetwork()
nn.add_layer(FullyConnectedLayer(neurons=3, activation='softmax'))

print('\n')
print('Training a Multi-Layer-Perceptron...')
# fit our nn
nn.fit(x, y, batch_size=1, epochs=5, learning_rate=.01, optimizer='rmsprop', loss='logloss',
       early_stopping=False)

print(y[0])
import numpy as np
print(nn.predict(np.array([[x[0]]])))






