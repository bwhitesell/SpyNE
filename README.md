<div align="center">
    <img src="https://raw.githubusercontent.com/bwhitesell/SpyNE/master/logo.png">
</div>

# SpyNE
SpyNE is a minimalist deep learning famework written 
"purely" in python. 

"proof" deep learning is easy. It's simplicty makes it an excellent tool to learn 
the mathematical framework behind deep learning.

SpyNE features a dead-simple api, super readable code,
no dependencies (other than numpy), eager execution by 
default and an accessible but exhaustive explanation of the 
underlying mathematical framework.

### Modules
SpyNE features custom implementations of the following:
- Automatic Differentiation (Reverse Mode)
- Optimization Engines
- An API to translate neural architectures into the AD 
  mini-language
  
Generally these are the fundamental components to all deep
learning frameworks.

### Installation (python 3.6+)
```
$ git clone https://github.com/bwhitesell/SpyNE
$ cd SpyNE
$ python setup.py install
```

### Example (Defining a Neural Network in 10 Seconds)
```python
from spyne.models import NeuralNetwork
from spyne.layers import FullyConnectedLayer

nn = NeuralNetwork()
nn.add_layer(FullyConnectedLayer(neurons=5, activation='relu', dropout=0))
nn.add_layer(FullyConnectedLayer(neurons=1, activation='linear'))
```
And that's it... we've defined a neural network architecture that has one 
hidden layer of 5 neurons per feature-axis and an ouput layer with a single 
neuron per feature-axis (as is desired for a regression problem).



Now let's get some data to train on...

```python
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
boston = load_boston()

# standardize the data
scaler = StandardScaler(with_mean=False)
x = scaler.fit_transform(boston['data'])
y = boston['target']
```

To train our neural net is just a single line.
```python
nn.fit(x, y, batch_size=10, epochs=20, learning_rate=.09, optimizer='rmsprop', l2=0.0000001)
```
The model performance will be printed as it is being trained:
```bash
Neural Network Architecture
Layers: 2
Params: 76 parameters
Using optimizer: RMSProp Optimizer


Batch 40/40, 100.0% for epoch 0:  Train Loss: 141.414 | Val Loss: 22.2118
Batch 40/40, 100.0% for epoch 1:  Train Loss: 96.3986 | Val Loss: 17.7557
Batch 40/40, 100.0% for epoch 2:  Train Loss: 74.4629 | Val Loss: 16.3234
Batch 40/40, 100.0% for epoch 3:  Train Loss: 61.1243 | Val Loss: 16.2161
Batch 40/40, 100.0% for epoch 4:  Train Loss: 51.9994 | Val Loss: 16.7056

```

### Project Status
```
Completed
- Reverse Mode Auto Diff
- Neural Net API
- SGD/Momentum/RMSProp Optimization
- L2 Regularization
- Fully Connected Layers
- Dropout

In Development
- Documentation
- Convolutional Layers (2d / 1d)
- Unit Test coverage up to MLP
    
Not Implemented
- Recurrent Layers (LSTM)
- GPU support (Probably through CuPy)
```
