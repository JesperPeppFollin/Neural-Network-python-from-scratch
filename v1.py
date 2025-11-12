import matplotlib.pyplot as plt
from keras.datasets import mnist

# Load MNIST dataset and divide into training/testing and features/labels
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Normalize pixel intensities to be between 0 and 1
train_X = train_X.astype('float32') / 255.0
test_X = test_X.astype('float32') / 255.0

# Visualize the first 5 images in the training set
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(train_X[i], cmap='gray')
    plt.title(f'Label: {train_y[i]}')
    plt.axis('off')
plt.show()