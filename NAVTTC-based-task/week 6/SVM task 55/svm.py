
import pandas as pd
from pathlib import Path

data_link = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"

col_names = ["variance", "skewness", "curtosis", "entropy", "class"]

# Download and load the original dataset
bankdata = pd.read_csv(
    data_link,
    names=col_names,
    sep=",",
    header=None
)

# Save the dataset as a separate CSV file
bankdata.to_csv("dataset.csv", index=False)

print("Dataset saved successfully!")
print(bankdata.head())
print("Dataset shape:", bankdata.shape)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the same banknote dataset used in the tutorial
data_link = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"

col_names = ["variance", "skewness", "curtosis", "entropy", "class"]

bankdata = pd.read_csv(
    data_link,
    names=col_names,
    sep=",",
    header=None
)

print("First five rows:")
print(bankdata.head())

print("\nDataset shape:", bankdata.shape)

# 2. Separate features and target
X = bankdata.drop("class", axis=1)
y = bankdata["class"]

# 3. Split data into training and testing sets
SEED = 42

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Train Linear SVM
linear_svm = SVC(kernel="linear")
linear_svm.fit(X_train, y_train)

# 5. Make predictions
linear_predictions = linear_svm.predict(X_test)

print("\n===== LINEAR SVM =====")
print("Accuracy:", accuracy_score(y_test, linear_predictions))
print("\nClassification Report:")
print(classification_report(y_test, linear_predictions))

# 6. Train Kernel SVM (RBF)
kernel_svm = SVC(kernel="rbf")
kernel_svm.fit(X_train, y_train)

kernel_predictions = kernel_svm.predict(X_test)

print("\n===== KERNEL SVM (RBF) =====")
print("Accuracy:", accuracy_score(y_test, kernel_predictions))
print("\nClassification Report:")
print(classification_report(y_test, kernel_predictions))

# 7. Compare confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(
    confusion_matrix(y_test, linear_predictions),
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[0]
)
axes[0].set_title("Linear SVM")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(
    confusion_matrix(y_test, kernel_predictions),
    annot=True,
    fmt="d",
    cmap="Greens",
    ax=axes[1]
)
axes[1].set_title("Kernel SVM (RBF)")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()
