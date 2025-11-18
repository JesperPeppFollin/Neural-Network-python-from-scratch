import numpy as np

class Math:

    # -----------------------------
    # Activations
    # -----------------------------
    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def deriv_sigmoid(Z):
        s = Math.sigmoid(Z)
        return s * (1 - s)

    @staticmethod
    def reLU(Z):
        return np.maximum(0, Z)

    @staticmethod
    def deriv_reLU(Z):
        return (Z > 0).astype(float)

    @staticmethod
    def tanh(Z):
        return np.tanh(Z)

    @staticmethod
    def deriv_tanh(Z):
        return 1 - np.tanh(Z)**2

    @staticmethod
    def softmax(Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    @staticmethod
    def linear(Z):
        return Z


    # -----------------------------
    # Loss functions
    # -----------------------------
    @staticmethod
    def cross_entropy(Y_pred, Y_true):
        m = Y_true.size
        Y_one_hot = Math.one_hot(Y_true, Y_pred.shape[0])
        return -np.sum(Y_one_hot * np.log(Y_pred + 1e-15)) / m

    @staticmethod
    def cross_entropy_grad(Y_pred, Y_true):
        m = Y_true.size
        Y_one_hot = Math.one_hot(Y_true, Y_pred.shape[0])
        return (Y_pred - Y_one_hot) / m

    @staticmethod
    def mse(Y_pred, Y_true):
        m = Y_true.size
        Y_one_hot = Math.one_hot(Y_true, Y_pred.shape[0])
        return np.sum((Y_pred - Y_one_hot)**2) / m

    @staticmethod
    def mse_grad(Y_pred, Y_true):
        m = Y_true.size
        Y_one_hot = Math.one_hot(Y_true, Y_pred.shape[0])
        return 2 * (Y_pred - Y_one_hot) / m


    # -----------------------------
    # One-hot encoding
    # -----------------------------
    @staticmethod
    def one_hot(Y, num_classes):
        one_hot_Y = np.zeros((num_classes, Y.size))
        one_hot_Y[Y, np.arange(Y.size)] = 1
        return one_hot_Y