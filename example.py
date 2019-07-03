import numpy as np

from models.graphs import NeuralNetwork, FullyConnectedLayer
from autodiff.variables.variables import Tensor, TensorConst

nn = NeuralNetwork()

x = Tensor([.1,.5,3,2,2,5,4])
l1 = FullyConnectedLayer(x, neurons=1, activation='')

nn.add_layer(l1)
y = TensorConst([1250])
print(y)
print('Learning relationship...')
for i in range(20):
    print('----- Pass -----')
    nn._learn_row(x, y)


