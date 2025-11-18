import numpy as np
from BaseClassificationNN import BaseClassificationNN
from Math import Math

class ClassificationNN(BaseClassificationNN):
    """
    Fully connected classification network.
    Uses forward/backprop logic based on your proven working example.
    """

    def forward(self, X):
        """
        Forward propagation through all layers.
        Stores linear combinations (Zs) and activations (As).
        """
        self.Zs = []
        self.As = [X]
        A = X

        for i in range(len(self.weights)):
            W, b = self.weights[i], self.biases[i]
            Z = np.dot(W, A) + b
            self.Zs.append(Z)

            # Hidden layers use chosen hidden activation
            if i < len(self.weights) - 1:
                A = getattr(Math, self.hidden_activation)(Z)
            else:
                # Output layer uses chosen output activation
                A = getattr(Math, self.output_activation)(Z)
            self.As.append(A)

        return A

    def backward(self, Y):
        """
        Backpropagation to compute gradients.
        Stores gradients in self.dWs and self.dbs.
        """
        m = Y.size
        Y_pred = self.As[-1]
        one_hot_Y = Math.one_hot(Y, self.num_classes)

        # Compute output gradient
        dZ = Y_pred - one_hot_Y

        self.dWs = [None] * len(self.weights)
        self.dbs = [None] * len(self.biases)

        # Backpropagate through all layers
        for i in reversed(range(len(self.weights))):
            A_prev = self.As[i]
            self.dWs[i] = np.dot(dZ, A_prev.T) / m
            self.dbs[i] = np.sum(dZ, axis=1, keepdims=True) / m

            if i > 0:
                W = self.weights[i]
                Z_prev = self.Zs[i-1]
                dZ = np.dot(W.T, dZ) * getattr(Math, f'deriv_{self.hidden_activation}')(Z_prev).astype(float)

    def train(self, X, Y, epochs=10, learning_rate=0.01, batch_size=64):
        """
        Simple training loop with mini-batches and SGD updates.
        """
        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[1])
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[permutation]

            for j in range(0, X.shape[1], batch_size):
                X_batch = X_shuffled[:, j:j+batch_size]
                Y_batch = Y_shuffled[j:j+batch_size]

                self.forward(X_batch)
                self.backward(Y_batch)
                self.update_params(self.dWs, self.dbs, learning_rate)

            # Evaluate accuracy after each epoch
            Y_pred = self.predict(X)
            acc = self.accuracy(Y_pred, Y)
            print(f"Epoch {epoch+1}/{epochs} - Accuracy: {acc:.4f}")

    def predict(self, X):
        """
        Predict labels for input X
        """
        Y_pred = self.forward(X)
        return np.argmax(Y_pred, axis=0)

    def accuracy(self, Y_pred, Y_true):
        """
        Compute classification accuracy
        """
        return np.sum(Y_pred == Y_true) / Y_true.size
