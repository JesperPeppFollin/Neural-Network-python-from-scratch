# the newtwork is a class now
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

class Math:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def reLU(Z):
        return np.maximum(0, Z)

    @staticmethod
    def deriv_reLU(Z):
        return Z > 0

    @staticmethod
    def softmax(Z):
        # Subtract max per sample for numerical stability
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    # detta är knappast en math funktion men okej
    @staticmethod
    def one_hot(Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y
    



class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            n_in  = layers[i]
            n_out = layers[i + 1]
            
            # He initialization (best for ReLU)
            W = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)
            b = np.zeros((n_out, 1))
            
            self.weights.append(W)
            self.biases.append(b)
            

nn = NeuralNetwork([784, 256, 10])