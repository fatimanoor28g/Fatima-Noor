
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Training data
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])

# 0 = Fail, 1 = Pass
model = LogisticRegression()
model.fit(X, y)

# Test data
X_test = np.array([[2], [5]])
predictions = model.predict(X_test)

print("Predictions:", predictions)
print("Accuracy:", accuracy_score([0, 1], predictions))

for hours, result in zip(X_test, predictions):
    status = "Pass" if result == 1 else "Fail"
    print(f"{hours[0]} study hours: {status}")
