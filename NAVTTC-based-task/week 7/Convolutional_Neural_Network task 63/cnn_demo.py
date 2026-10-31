
import numpy as np

# A small 5x5 grayscale image
image = np.array([
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1]
], dtype=float)

# A 3x3 convolution filter
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=float)

# Apply convolution (valid, stride 1)
feature_map = np.zeros((3, 3))

for i in range(3):
    for j in range(3):
        region = image[i:i+3, j:j+3]
        feature_map[i, j] = np.sum(region * kernel)

# ReLU activation
feature_map = np.maximum(0, feature_map)

# 2x2 max pooling with stride 1
pooled = np.zeros((2, 2))

for i in range(2):
    for j in range(2):
        region = feature_map[i:i+2, j:j+2]
        pooled[i, j] = np.max(region)

print("Input Image:")
print(image)

print("\nConvolution Filter:")
print(kernel)

print("\nFeature Map after ReLU:")
print(feature_map)

print("\nMax Pooling Output:")
print(pooled)
