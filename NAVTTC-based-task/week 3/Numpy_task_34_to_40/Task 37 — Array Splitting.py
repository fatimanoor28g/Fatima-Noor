
import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60])

parts = np.array_split(arr, 3)

print("Original array:", arr)
print("Split arrays:")

for part in parts:
    print(part)

arr2 = np.array([[1, 2, 3], [4, 5, 6]])

print("Split along columns:")
for part in np.array_split(arr2, 3, axis=1):
    print(part)
