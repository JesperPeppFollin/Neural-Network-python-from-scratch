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

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
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
    W1 = np.random.randn(256, 784) * np.sqrt(1/784) # 784 pixels in each image
    b1 = np.zeros((256, 1))
    # hidden layer 2
    W2 = np.random.randn(10, 256) * np.sqrt(1/256)
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
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_reLU(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2
    
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2

# Load MNIST dataset and prepare like the YouTube example
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Combine features and labels, then shuffle
m_total = X_train.shape[0]
X_flat = X_train.reshape(m_total, -1)  # (60000, 784)
data = np.column_stack((y_train, X_flat))  # (60000, 785) - label in first column
np.random.shuffle(data)

# Split into dev and training sets
data_dev = data[0:1000].T  # (785, 1000)
Y_dev = data_dev[0].astype(int)  # (1000,)
X_dev = data_dev[1:785]  # (784, 1000)
X_dev = X_dev / 255.0

data_train = data[1000:m_total].T  # (785, 59000)
Y_train = data_train[0].astype(int)  # (59000,)
X_train = data_train[1:785]  # (784, 59000)
X_train = X_train / 255.0

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 100)


# using matplotlib to visualize examples of where model is failing
Z1_dev, A1_dev, Z2_dev, A2_dev = forward_prop(X_dev, W1, b1, W2, b2)
predictions_dev = get_predictions(A2_dev)
incorrect_indices = np.where(predictions_dev != Y_dev)[0]
num_incorrect_to_show = min(5, len(incorrect_indices))
plt.figure(figsize=(10, 2))
for i in range(num_incorrect_to_show):
    idx = incorrect_indices[i]
    plt.subplot(1, num_incorrect_to_show, i + 1)
    plt.imshow(X_dev[:, idx].reshape(28, 28), cmap='gray')
    plt.title(f'True: {Y_dev[idx]}, Pred: {predictions_dev[idx]}')
    plt.axis('off')
plt.show()