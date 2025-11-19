this is a good start for a more flexible NN.
For even more flexibility and better overall implementation I should take some inspiration from pyTorch and make init something like 
--> instanceOfModelClass = modelClass(
    layerType(number of input neurons, number of output neurons activation function)
    layerType(number of input neurons, number of output neurons activation function)
    layerType(number of input neurons, number of output neurons activation function)
    ... continue to add more layers
)

layerType could be base or convolutional etc