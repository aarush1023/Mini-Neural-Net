import numpy as np
from tensorflow.keras.datasets import mnist


class MiniMNISTNeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10, learning_rate=0.1):
        print("Initializing neural network...")
        print(f"Input size: {input_size}")
        print(f"Hidden layer size: {hidden_size}")
        print(f"Output size: {output_size}")
        print(f"Learning rate: {learning_rate}")

        self.learning_rate = learning_rate

        print("Creating weights and biases...")

        # He initialization for ReLU
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        print("Weights and biases created.")
        print(f"W1 shape: {self.W1.shape}")
        print(f"b1 shape: {self.b1.shape}")
        print(f"W2 shape: {self.W2.shape}")
        print(f"b2 shape: {self.b2.shape}")
        print()

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return x > 0

    def softmax(self, x):
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

        clipped = np.clip(predictions, 1e-9, 1 - 1e-9)
        correct_log_probs = -np.log(clipped[range(m), y_true])

        return np.mean(correct_log_probs)

    def backward(self, X, y_true):
        m = X.shape[0]

        # Softmax + cross-entropy gradient
        dz2 = self.a2.copy()
        dz2[range(m), y_true] -= 1
        dz2 /= m

        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=64):
        n_samples = X_train.shape[0]
        num_batches = int(np.ceil(n_samples / batch_size))

        print("Starting training...")
        print(f"Training samples: {n_samples}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Batches per epoch: {num_batches}")
        print()

        for epoch in range(epochs):
            print("=" * 60)
            print(f"Starting epoch {epoch + 1}/{epochs}")
            print("=" * 60)

            indices = np.random.permutation(n_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]

            print("Data shuffled.")

            for batch_number, start in enumerate(range(0, n_samples, batch_size), start=1):
                end = start + batch_size

                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                predictions = self.forward(X_batch)
                self.backward(X_batch, y_batch)

                # Print progress every 100 batches
                if batch_number % 100 == 0 or batch_number == 1 or batch_number == num_batches:
                    batch_loss = self.cross_entropy_loss(predictions, y_batch)

                    print(
                        f"Epoch {epoch + 1}, "
                        f"Batch {batch_number}/{num_batches}, "
                        f"Batch loss: {batch_loss:.4f}"
                    )

            print("Epoch training complete. Evaluating...")

            train_predictions = self.forward(X_train[:5000])
            train_loss = self.cross_entropy_loss(train_predictions, y_train[:5000])

            test_accuracy = self.accuracy(X_test, y_test)

            print()
            print(f"Epoch {epoch + 1} summary:")
            print(f"Training loss on 5,000 samples: {train_loss:.4f}")
            print(f"Test accuracy: {test_accuracy * 100:.2f}%")
            print()

        print("Training finished.")
        print()

    def predict(self, X):
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


# -----------------------------
# Load MNIST data
# -----------------------------

print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("MNIST loaded.")
print(f"Original X_train shape: {X_train.shape}")
print(f"Original y_train shape: {y_train.shape}")
print(f"Original X_test shape: {X_test.shape}")
print(f"Original y_test shape: {y_test.shape}")
print()

# -----------------------------
# Prepare data
# -----------------------------

print("Flattening images...")
print("Each image is 28 x 28 pixels.")
print("28 x 28 = 784 input values.")

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

print(f"Flattened X_train shape: {X_train.shape}")
print(f"Flattened X_test shape: {X_test.shape}")
print()

print("Normalizing pixel values...")
print("Before normalization, pixels range from 0 to 255.")
print("After normalization, pixels range from 0 to 1.")

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

print(f"Training pixel min: {X_train.min():.2f}")
print(f"Training pixel max: {X_train.max():.2f}")
print()

# -----------------------------
# Create model
# -----------------------------

model = MiniMNISTNeuralNetwork(
    input_size=784,
    hidden_size=128,
    output_size=10,
    learning_rate=0.1
)

# -----------------------------
# Train model
# -----------------------------

model.train(
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=10,
    batch_size=64
)

# -----------------------------
# Make predictions
# -----------------------------

print("Testing the model on the first 10 test images...")

sample_images = X_test[:10]
sample_labels = y_test[:10]

predictions = model.predict(sample_images)

print()
print("Predictions vs actual labels:")
for i in range(10):
    print(f"Image {i + 1}: predicted {predictions[i]}, actual {sample_labels[i]}")

final_accuracy = model.accuracy(X_test, y_test)

print()
print(f"Final test accuracy: {final_accuracy * 100:.2f}%")