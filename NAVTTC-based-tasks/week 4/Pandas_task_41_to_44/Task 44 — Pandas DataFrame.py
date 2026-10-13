
import pandas as pd

data = {
    "Name": ["Ali", "Sara", "Ahmed", "Fatima"],
    "Age": [20, 21, 19, 22],
    "Marks": [85, 90, 78, 95]
}

df = pd.DataFrame(data)

print("Complete DataFrame:")
print(df)

print("\nName column:")
print(df["Name"])

print("\nStudents with marks above 80:")
print(df[df["Marks"] > 80])

print("\nDataFrame shape:", df.shape)
