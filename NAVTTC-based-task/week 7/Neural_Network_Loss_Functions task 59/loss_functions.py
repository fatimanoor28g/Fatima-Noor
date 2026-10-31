
import numpy as np

# Actual values
y_true = np.array([1.0, 0.0, 1.0, 0.0])

# Predicted values
y_pred = np.array([0.8, 0.2, 0.9, 0.1])

# Mean Squared Error
mse = np.mean((y_true - y_pred) ** 2)

# Mean Absolute Error
mae = np.mean(np.abs(y_true - y_pred))

# Binary Cross-Entropy
epsilon = 1e-7
y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

binary_cross_entropy = -np.mean(
    y_true * np.log(y_pred_clipped)
    + (1 - y_true) * np.log(1 - y_pred_clipped)
)

print("Actual values:", y_true)
print("Predicted values:", y_pred)

print("\nMean Squared Error:", round(mse, 4))
print("Mean Absolute Error:", round(mae, 4))
print("Binary Cross-Entropy:", round(binary_cross_entropy, 4))
