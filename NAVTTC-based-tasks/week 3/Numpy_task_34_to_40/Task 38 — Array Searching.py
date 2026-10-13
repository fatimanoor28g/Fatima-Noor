
import numpy as np

arr = np.array([10, 20, 30, 20, 40, 20])

result = np.where(arr == 20)
print("Positions of 20:", result[0])

result2 = np.where(arr > 25)
print("Positions of values greater than 25:", result2[0])

result3 = np.searchsorted(arr, 25, sorter=np.argsort(arr))
print("Position to insert 25 in sorted order:", result3)
