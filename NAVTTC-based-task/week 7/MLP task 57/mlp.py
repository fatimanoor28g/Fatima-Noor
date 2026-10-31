
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Input data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Expected output
y = np.array([0, 1, 1, 0, 0, 1, 1, 0])

# Create the MLP model
model = MLPClassifier(
    hidden_layer_sizes=(5,),
    activation="relu",
    solver="lbfgs",
    max_iter=2000,
    random_state=42
)

# Train the model
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

print("MLP Neural Network")
print("------------------")
print("Input data:")
print(X)

print("\nActual outputs:")
print(y)

print("\nPredicted outputs:")
print(predictions)

print("\nAccuracy:")
print(accuracy_score(y, predictions) * 100, "%")
