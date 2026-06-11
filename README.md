# MNIST Neural Network From Scratch

## Overview

This project builds a simple neural network from scratch to classify handwritten digits from the MNIST dataset.

The model is written mostly with NumPy and does not use TensorFlow or PyTorch for training. TensorFlow/Keras is only used to load the MNIST dataset.

The goal of this project is to understand how a basic neural network works internally, including forward propagation, backpropagation, gradient descent, activation functions, and loss calculation.

---

## What the Project Does

The program trains a neural network to recognize handwritten digits from `0` to `9`.

Each MNIST image is:

```text
28 x 28 pixels
```

The image is flattened into:

```text
784 input values
```

The neural network architecture is:

```text
784 inputs → 128 hidden neurons → 10 outputs
```

The final output is a prediction for which digit the image represents.

---

## Features

- Loads the MNIST handwritten digit dataset
- Converts each `28 x 28` image into a flat vector of 784 values
- Normalizes pixel values from `0–255` to `0–1`
- Implements a fully connected neural network
- Uses ReLU activation in the hidden layer
- Uses softmax activation in the output layer
- Uses cross-entropy loss
- Uses mini-batch gradient descent
- Implements backpropagation manually
- Prints progress during training
- Shows predictions on sample test images
- Reports final test accuracy

---

## Requirements

Install the required Python packages:

```bash
pip install numpy tensorflow
```

TensorFlow is only used for loading the MNIST dataset. The neural network itself is trained using NumPy.

---

## How to Run

Save the code in a file named:

```bash
mnist_neural_network.py
```

Then run:

```bash
python mnist_neural_network.py
```

---

## Expected Output

When the program runs, it prints progress messages such as:

```text
Loading MNIST dataset...
MNIST loaded.
Original X_train shape: (60000, 28, 28)
Original y_train shape: (60000,)

Flattening images...
Each image is 28 x 28 pixels.
28 x 28 = 784 input values.

Initializing neural network...
Input size: 784
Hidden layer size: 128
Output size: 10

Starting training...
Training samples: 60000
Test samples: 10000
Epochs: 10
Batch size: 64
Batches per epoch: 938
```

During training, it prints batch loss and epoch summaries:

```text
Epoch 1, Batch 1/938, Batch loss: 2.3901
Epoch 1, Batch 100/938, Batch loss: 0.7562

Epoch 1 summary:
Training loss on 5,000 samples: 0.4123
Test accuracy: 88.90%
```

After several epochs, the model should usually reach around:

```text
92%–96% test accuracy
```

The exact result may vary because the model weights are initialized randomly.

---

## How the Neural Network Works

The model learns through repeated training steps.

For each mini-batch:

1. Images are passed into the network.
2. The hidden layer applies a weighted transformation and ReLU activation.
3. The output layer produces probabilities for each digit using softmax.
4. The cross-entropy loss measures how wrong the prediction is.
5. Backpropagation calculates how each weight contributed to the error.
6. Gradient descent updates the weights and biases.

Over time, the network improves its predictions.

---

## Main Concepts Used

### ReLU Activation

ReLU stands for Rectified Linear Unit.

```python
relu(x) = max(0, x)
```

It helps the network learn nonlinear patterns.

---

### Softmax

Softmax converts raw output scores into probabilities.

For example, the output may look like:

```text
[0.01, 0.02, 0.91, 0.01, 0.00, 0.02, 0.01, 0.00, 0.01, 0.01]
```

This means the model is most confident that the image is the digit `2`.

---

### Cross-Entropy Loss

Cross-entropy loss measures how far the predicted probabilities are from the correct answer.

Lower loss means the model is improving.

---

### Backpropagation

Backpropagation calculates how much each weight and bias should change to reduce the loss.

This is the main learning mechanism of the neural network.

---

## Suggested File Structure

```text
mnist-neural-network/
│
├── README.md
└── mnist_neural_network.py
```

---

## Possible Improvements

Future improvements could include:

- Add another hidden layer
- Increase the hidden layer size
- Add dropout
- Add learning-rate decay
- Plot training loss over time
- Save and load trained weights
- Add a validation set
- Build a convolutional neural network

---

## Educational Purpose

This project is intended to demonstrate how a neural network works under the hood.

Instead of relying on a machine learning framework to handle training automatically, this project manually implements the core steps of neural network learning using NumPy.
