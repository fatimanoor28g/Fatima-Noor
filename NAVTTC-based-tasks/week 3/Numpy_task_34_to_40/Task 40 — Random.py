
import numpy as np

print("Random integer:", np.random.randint(1, 100))

print("Random decimal:", np.random.rand())

print("Three random integers:", np.random.randint(1, 50, size=3))

print("Random array:")
print(np.random.randint(1, 10, size=(2, 3)))

print("Random choice:", np.random.choice([10, 20, 30, 40]))
