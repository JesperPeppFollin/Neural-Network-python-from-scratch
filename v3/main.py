from v3.VanillaClassificationNN import ClassificationNN
from keras.datasets import mnist, fashion_mnist


# load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Prepare training data
m_train = X_train.shape[0]
X_train_flat = X_train.reshape(m_train, -1).T / 255.0
Y_train = y_train.astype(int)

# Prepare test data (separate, unseen data)
m_test = X_test.shape[0]
X_test_flat = X_test.reshape(m_test, -1).T / 255.0
Y_test = y_test.astype(int)

model = ClassificationNN(
    layers=[784, 128, 64, 10],
    hidden_activation='reLU',
    output_activation='softmax',
    loss='cross_entropy'
)

# train and test model
model.train(X_train_flat, Y_train, epochs=10, learning_rate=0.01, batch_size=64)
Y_pred_test = model.predict(X_test_flat)
accuracy = model.accuracy(Y_pred_test, Y_test)
print(f"Test Accuracy: {accuracy:.4f}")