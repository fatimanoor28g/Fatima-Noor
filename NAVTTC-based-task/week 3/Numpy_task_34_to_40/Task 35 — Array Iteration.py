
import numpy as np

arr = np.array([10, 20, 30, 40])

print("1D array elements:")
for item in arr:
    print(item)

arr2 = np.array([[1, 2, 3], [4, 5, 6]])

print("2D array elements:")
for row in arr2:
    for item in row:
        print(item)

print("Using nditer:")
for item in np.nditer(arr2):
    print(item)
