
import pandas as pd

marks = pd.Series([85, 90, 78, 95],
                  index=["Ali", "Sara", "Ahmed", "Fatima"])

print("Student Marks:")
print(marks)

print("\nSara's marks:", marks["Sara"])
print("Maximum marks:", marks.max())
print("Average marks:", marks.mean())
