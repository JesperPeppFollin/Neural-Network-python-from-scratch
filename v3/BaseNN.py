from abc import ABC, abstractmethod
import numpy as np

class BaseNN(ABC):
    
    def __init__(self, layers):
        
        self.layers = layers # List defining the number of layers and number of neurons in each layer
        self.weights = [] # List to hold weights for each layer
        self.biases = [] # List to hold biases for each layer
        self.Zs = None # Linear combination per layer
        self.As = None # Activation per layer
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            n_in = layers[i]
            n_out = layers[i + 1]

            W = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in) # He initialization (good for ReLU, can be overridden in children classes)
            b = np.zeros((n_out, 1))
            self.weights.append(W)
            self.biases.append(b)
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass through the network.
        Must be implemented by the child class, which will apply activations.
        """
        pass

    @abstractmethod
    def backward(self, dLoss):
        """
        Backward pass / gradient computation.
        Must be implemented by the child class.
        Should return gradients: dWs, dbs
        """
        pass

    def update_params(self, dWs, dbs, learning_rate):
        """Default SGD update; can be overridden by children"""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dWs[i]
            self.biases[i] -= learning_rate * dbs[i]
    
    
    # Optional: leave these abstract(?) or implement in task-specific base classes
    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def accuracy(self, Y_pred, Y_true):
        pass