
import numpy as np

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([1.5, 2.5, 3.5])
arr3 = np.array([1, 2, 3], dtype="float")

print("Integer array:", arr1)
print("Integer data type:", arr1.dtype)

print("Float array:", arr2)
print("Float data type:", arr2.dtype)

print("Converted array:", arr3)
print("Converted data type:", arr3.dtype)
