this is a good start for a more flexible NN.
For even more flexibility and better overall implementation I should take some inspiration from pyTorch and make init something like 
--> nn = baseModel(
    layerType(number of neurons, activation function)
    layerType(number of neurons, activation function)
    layerType(number of neurons, activation function)
)

layerType could be base or convolutional etc