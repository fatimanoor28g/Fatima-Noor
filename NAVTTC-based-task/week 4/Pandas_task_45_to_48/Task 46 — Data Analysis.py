
import pandas as pd

data = {
    "Name": ["Ali", "Sara", "Ahmed", "Fatima"],
    "Age": [20, 21, 19, 22],
    "Marks": [85, 90, 78, 95]
}

df = pd.DataFrame(data)

print("First five rows:")
print(df.head())

print("\nStatistical summary:")
print(df.describe())

print("\nAverage marks:", df["Marks"].mean())
print("Highest marks:", df["Marks"].max())
print("Lowest marks:", df["Marks"].min())
