
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the dataset
data = pd.read_csv("diabetes.csv", header=None)
data.columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

print("First 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

# 2. Separate features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# 4. Create and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Predict for one sample
sample = X_test.iloc[[0]]
prediction = model.predict(sample)

print("\nSample Features:")
print(sample)

print("Predicted Outcome:", prediction[0])
print("Actual Outcome:", y_test.iloc[0])
