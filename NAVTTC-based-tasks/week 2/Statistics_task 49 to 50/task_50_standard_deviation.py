
# Task 50: Standard Deviation

import statistics

numbers = [10, 12, 23, 23, 16, 23, 21, 16]

standard_deviation = statistics.stdev(numbers)

print("Numbers:", numbers)
print("Standard Deviation:", round(standard_deviation, 2))
