
import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

joined = np.concatenate((arr1, arr2))
print("Joined 1D arrays:", joined)

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

vertical = np.concatenate((a, b), axis=0)
horizontal = np.concatenate((a, b), axis=1)

print("Vertical joining:\n", vertical)
print("Horizontal joining:\n", horizontal)
