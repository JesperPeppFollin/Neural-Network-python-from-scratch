from abc import abstractmethod
import numpy as np

# CHANGE THESE TO USE PACKAGE INSTEAD
from Math import Math
from BaseNN import BaseNN

class BaseClassificationNN(BaseNN):
    """
    Base class for classification neural networks.
    Provides:
    - Hidden/output activation selection
    - Classification loss function selection
    - Loss gradients
    - Leaves train/predict to be implemented by children
    """
    
    # add more allowed activations and losses here if needed
    _hidden_activations = ['reLU', 'sigmoid', 'tanh']
    _output_activations = ['softmax', 'sigmoid', 'linear']
    _losses = ['cross_entropy', 'mse']
    
    def __init__(self, layers, hidden_activation='reLU', output_activation='softmax', loss='cross_entropy'):
        super().__init__(layers)
        self.num_classes = layers[-1]
        
        # Hidden activation initialization
        if hidden_activation not in self._hidden_activations:
            raise ValueError(f"Hidden activation must be one of {self._hidden_activations}")
        self.hidden_activation = hidden_activation

        # Output activation initialization
        if output_activation not in self._output_activations:
            raise ValueError(f"Output activation must be one of {self._output_activations}")
        self.output_activation = output_activation

        # Loss initialization
        if loss not in self._losses:
            raise ValueError(f"Loss must be one of {self._losses}")
        self.loss_name = loss
        

    # ------------------------
    # Activation functions
    # ------------------------
    def activation(self, Z, layer_index):
        """Returns activation for the layer"""
        if layer_index == len(self.weights) - 1:
            return getattr(Math, self.output_activation)(Z)
        else:
            return getattr(Math, self.hidden_activation)(Z)

    def activation_derivative(self, Z, layer_index):
        """Returns derivative for backpropagation"""
        if layer_index == len(self.weights) - 1:
            return np.ones_like(Z)  # Output handled in loss gradient
        else:
            return getattr(Math, f'deriv_{self.hidden_activation}')(Z).astype(float)
        

    # ------------------------
    # Loss functions
    # ------------------------
    def compute_loss(self, Y_pred, Y_true):
        """Compute scalar loss"""
        return getattr(Math, self.loss_name)(Y_pred, Y_true)

    def compute_loss_grad(self, Y_pred, Y_true):
        """Compute gradient of loss w.r.t output"""
        return getattr(Math, f'{self.loss_name}_grad')(Y_pred, Y_true)
    
    
    # ------------------------
    # Abstract methods for child
    # ------------------------
    @abstractmethod
    def train(self, *args, **kwargs):
        """Child implements training loop (batching, optimizer, etc.)"""
        pass

    @abstractmethod
    def predict(self, X):
        """Child implements how to produce predictions"""
        pass
    
    @abstractmethod
    def accuracy(self, Y_pred, Y_true):
        """Child implements how to produce accuracy"""
        pass