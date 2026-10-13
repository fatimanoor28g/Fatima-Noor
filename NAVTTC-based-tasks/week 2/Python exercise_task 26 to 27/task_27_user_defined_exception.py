
# Task 27: User-Defined Exception

class InvalidAgeError(Exception):
    pass


def check_age(age):
    if age < 0:
        raise InvalidAgeError("Age cannot be negative.")

    elif age < 18:
        print("You are a minor.")

    else:
        print("You are an adult.")


try:
    age = int(input("Enter your age: "))
    check_age(age)

except InvalidAgeError as error:
    print("Custom Error:", error)

except ValueError:
    print("Error: Please enter a valid integer.")
