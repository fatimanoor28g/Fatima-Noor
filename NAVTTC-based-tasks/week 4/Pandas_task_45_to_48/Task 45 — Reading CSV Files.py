
import pandas as pd

data = {
    "Name": ["Ali", "Sara", "Ahmed", "Fatima"],
    "Age": [20, 21, 19, 22],
    "Marks": [85, 90, 78, 95]
}

df = pd.DataFrame(data)
df.to_csv("students.csv", index=False)

loaded_df = pd.read_csv("students.csv")

print("CSV file loaded:")
print(loaded_df)
