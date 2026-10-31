
import numpy as np

# Input data: 3 samples, 2 features each
X = np.array([
    [0.5, 0.2],
    [0.8, 0.6],
    [0.1, 0.9]
])

# Weights and biases for hidden layer
W1 = np.array([
    [0.4, 0.7],
    [0.3, 0.2]
])

b1 = np.array([0.1, 0.1])

# Weights and bias for output layer
W2 = np.array([
    [0.6],
    [0.5]
])

b2 = np.array([0.2])

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward propagation
hidden_input = np.dot(X, W1) + b1
hidden_output = relu(hidden_input)

final_input = np.dot(hidden_output, W2) + b2
final_output = sigmoid(final_input)

print("Input Data:")
print(X)

print("\nHidden Layer Output:")
print(hidden_output)

print("\nFinal Output:")
print(final_output)
