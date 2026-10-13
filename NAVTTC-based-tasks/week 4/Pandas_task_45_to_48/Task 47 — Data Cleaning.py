
import pandas as pd

data = {
    "Name": ["Ali", "Sara", None, "Fatima", "Ali"],
    "Age": [20, 21, 19, None, 20],
    "Marks": [85, 90, 78, 95, 85]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

df["Name"] = df["Name"].fillna("Unknown")
df["Age"] = df["Age"].fillna(df["Age"].mean())
df = df.drop_duplicates()

print("\nCleaned Data:")
print(df)
