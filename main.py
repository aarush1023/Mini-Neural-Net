import numpy as np
from tensorflow.keras.datasets import mnist


class MiniMNISTNeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10, learning_rate=0.1):
        self.learning_rate = learning_rate

        # He initialization for ReLU
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return x > 0

    def softmax(self, x):
        # subtract max for numerical stability
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2

    def cross_entropy_loss(self, predictions, y_true):
        m = y_true.shape[0]

        # avoid log(0)
        clipped = np.clip(predictions, 1e-9, 1 - 1e-9)

        correct_log_probs = -np.log(clipped[range(m), y_true])
        loss = np.mean(correct_log_probs)

        return loss

    def backward(self, X, y_true):
        m = X.shape[0]

        # Convert softmax output into gradient
        dz2 = self.a2.copy()
        dz2[range(m), y_true] -= 1
        dz2 /= m

        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Gradient descent update
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=64):
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size

                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                predictions = self.forward(X_batch)
                self.backward(X_batch, y_batch)

            # Evaluate after each epoch
            train_predictions = self.forward(X_train[:5000])
            train_loss = self.cross_entropy_loss(train_predictions, y_train[:5000])
            test_accuracy = self.accuracy(X_test, y_test)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Test Accuracy: {test_accuracy:.4f}"
            )

    def predict(self, X):
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


# -----------------------------
# Load MNIST data
# -----------------------------

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten 28x28 images into 784-dimensional vectors
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Normalize pixel values from 0–255 to 0–1
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# -----------------------------
# Train model
# -----------------------------

model = MiniMNISTNeuralNetwork(
    input_size=784,
    hidden_size=128,
    output_size=10,
    learning_rate=0.1
)

model.train(
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=10,
    batch_size=64
)

# -----------------------------
# Try predictions
# -----------------------------

sample_images = X_test[:10]
sample_labels = y_test[:10]

predictions = model.predict(sample_images)

print("\nPredictions:", predictions)
print("Actual:     ", sample_labels)