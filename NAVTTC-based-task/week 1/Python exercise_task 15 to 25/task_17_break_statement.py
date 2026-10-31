
# Task 17: Break Statement

# Stop the loop when number reaches 4
for i in range(1, 11):
    if i == 4:
        break
    print(i)

# Search for a number
numbers = [10, 20, 30, 40, 50]
for num in numbers:
    if num == 30:
        print("Number found:", num)
        break
