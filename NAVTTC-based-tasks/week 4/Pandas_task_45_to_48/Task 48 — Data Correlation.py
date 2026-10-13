
import pandas as pd

data = {
    "Study_Hours": [1, 2, 3, 4, 5],
    "Marks": [40, 50, 65, 75, 90],
    "Sleep_Hours": [8, 7, 7, 6, 6]
}

df = pd.DataFrame(data)

print("Student Data:")
print(df)

print("\nCorrelation Matrix:")
print(df.corr())

print("\nStudy Hours and Marks Correlation:")
print(df["Study_Hours"].corr(df["Marks"]))
