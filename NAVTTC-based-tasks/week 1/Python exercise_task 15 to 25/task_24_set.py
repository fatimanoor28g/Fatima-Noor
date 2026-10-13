
# Task 24: Set

numbers = {1, 2, 3, 4, 4}

# Display set (duplicate values are removed)
print("Set:", numbers)

# Add an item
numbers.add(5)
print("After add:", numbers)

# Remove an item
numbers.remove(2)
print("After remove:", numbers)

# Set operations
a = {1, 2, 3}
b = {3, 4, 5}

print("Union:", a.union(b))
print("Intersection:", a.intersection(b))
print("Difference:", a.difference(b))
