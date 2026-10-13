
# Task 26: Exception Handling

# Example 1: Division
try:
    a = int(input("Enter first number: "))
    b = int(input("Enter second number: "))

    result = a / b
    print("Division result:", result)

except ZeroDivisionError:
    print("Error: Cannot divide by zero.")

except ValueError:
    print("Error: Please enter valid integers.")

finally:
    print("Division program completed.")


# Example 2: List index
numbers = [10, 20, 30]

try:
    index = int(input("Enter list index (0 to 2): "))
    print("List value:", numbers[index])

except IndexError:
    print("Error: Index is out of range.")

except ValueError:
    print("Error: Please enter a valid integer.")
