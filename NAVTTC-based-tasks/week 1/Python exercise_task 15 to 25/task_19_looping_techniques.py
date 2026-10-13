
# Task 19: Looping Techniques

# 1. Using enumerate()
fruits = ["Apple", "Mango", "Banana"]
for index, fruit in enumerate(fruits):
    print(index, fruit)

# 2. Using zip()
names = ["Ali", "Sara", "Ahmed"]
marks = [80, 90, 75]
for name, mark in zip(names, marks):
    print(name, mark)

# 3. Using range()
for i in range(2, 11, 2):
    print(i)

# 4. Using reversed()
for fruit in reversed(fruits):
    print(fruit)

# 5. Using sorted()
numbers = [5, 2, 8, 1, 4]
for num in sorted(numbers):
    print(num)
