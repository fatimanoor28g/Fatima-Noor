
import pandas as pd

data = {
    "Name": ["Ali", "Sara", "Ahmed", "Fatima"],
    "Age": [20, 21, 19, 22],
    "Marks": [85, 90, 78, 95]
}

df = pd.DataFrame(data)

print("Student Data:")
print(df)

print("\nFirst two rows:")
print(df.head(2))

print("\nData information:")
print(df.info())
