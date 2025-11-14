# everything hard coded, no room for adjustments, just to get something working first

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist


def reLU(Z):
    return np.maximum(0, Z)


def deriv_reLU(Z):
    return Z > 0


def softmax(Z):
    # Subtract max per sample for numerical stability
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)


def one_hot(Y, nbr_classes=10):
    one_hot_Y = np.zeros((Y.size, nbr_classes))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


# NN_strucutre = [784, 16, 16, 10] något sånt här sen kanske?


def init_params():
    # hidden layer 1
    W1 = np.random.randn(300, 784) * np.sqrt(2 / 784)  # 784 pixels in each image
    b1 = np.zeros((300, 1))
    # hidden layer 2
    W2 = np.random.randn(10, 300) * np.sqrt(2 / 300)
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2


def forward_prop(X, W1, b1, W2, b2):
    Z1 = W1.dot(X) + b1
    A1 = reLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_reLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    batch_size = 64

    for i in range(iterations):

        # Shuffle dataset at the start of each epoch
        permutation = np.random.permutation(X.shape[1])
        X = X[:, permutation]
        Y = Y[permutation]

        # Loop over mini-batches
        for j in range(0, X.shape[1], batch_size):
            X_batch = X[:, j : j + batch_size]
            Y_batch = Y[j : j + batch_size]

            Z1, A1, Z2, A2 = forward_prop(X_batch, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X_batch, Y_batch)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 1 == 0:
            _, _, _, A2_full = forward_prop(X, W1, b1, W2, b2)
            print("Epoch:", i, "Accuracy:", get_accuracy(get_predictions(A2_full), Y))

    return W1, b1, W2, b2

def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(X, W1, b1, W2, b2)
    predictions = get_predictions(A2)
    return predictions



# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Prepare training data
m_train = X_train.shape[0]
X_train_flat = X_train.reshape(m_train, -1).T / 255.0  # (784, 60000)
Y_train = y_train.astype(int)  # (60000,)

# Prepare test data (separate, unseen data)
m_test = X_test.shape[0]
X_test_flat = X_test.reshape(m_test, -1).T / 255.0  # (784, 10000)
Y_test = y_test.astype(int)  # (10000,)

W1, b1, W2, b2 = gradient_descent(X_train_flat, Y_train, 0.01, 10)

predictions = predict(X_test_flat, W1, b1, W2, b2)
print("Final test set accuracy:", get_accuracy(predictions, Y_test))



# incorrect_indices = np.where(predictions != Y_test)[0]
# num_incorrect_to_show = min(5, len(incorrect_indices))
# plt.figure(figsize=(10, 2))
# for i in range(num_incorrect_to_show):
#     idx = incorrect_indices[i]
#     plt.subplot(1, num_incorrect_to_show, i + 1)
#     plt.imshow(X_test_flat[:, idx].reshape(28, 28), cmap='gray')
#     plt.title(f'True: {Y_test[idx]}, Pred: {predictions[idx]}')
#     plt.axis('off')
# plt.show()
