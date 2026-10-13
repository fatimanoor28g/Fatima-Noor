
import numpy as np

original = np.array([10, 20, 30, 40])

copied = original.copy()
viewed = original.view()

original[0] = 100

print("Original array:", original)
print("Copied array:", copied)
print("Viewed array:", viewed)

print("Copy shares data:", np.shares_memory(original, copied))
print("View shares data:", np.shares_memory(original, viewed))
