
# Task 25: Dictionary

student = {
    "name": "Ali",
    "age": 20,
    "marks": 85
}

# Display dictionary
print("Student:", student)

# Access values
print("Name:", student["name"])
print("Marks:", student["marks"])

# Add a new key-value pair
student["city"] = "Lahore"
print("After adding city:", student)

# Update a value
student["marks"] = 90
print("Updated marks:", student["marks"])

# Loop through dictionary
for key, value in student.items():
    print(key, ":", value)

# Display keys and values
print("Keys:", list(student.keys()))
print("Values:", list(student.values()))
