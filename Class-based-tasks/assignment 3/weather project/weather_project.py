import csv
import math

# ------------------------------------------------------------
# STEP 1: LOAD WEATHER DATA
# ------------------------------------------------------------

rows = []

with open("weather_data.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        rows.append(row)

print("Total weather records loaded:", len(rows))
print()


# ------------------------------------------------------------
# STEP 2: INITIAL WEIGHTS AND BIASES
# ------------------------------------------------------------

w11 = 0.5
w12 = 0.2
w13 = 0.4
b1 = -0.1

w21 = -0.3
w22 = 0.8
w23 = 0.1
b2 = 0.05

v1 = 1.0
v2 = 0.7
b3 = -0.2

learning_rate = 0.1
epochs = 1000


# ------------------------------------------------------------
# STEP 3: TRAINING
# ------------------------------------------------------------

for epoch in range(epochs):

    total_loss = 0

    for row in rows:

        # Normalize inputs
        x1 = float(row["temperature_c"]) / 100
        x2 = float(row["humidity_percent"]) / 100
        x3 = (float(row["pressure_hpa"]) - 1000) / 20

        y = int(row["rain_next_hour"])

        # -------------------------
        # FORWARD PASS
        # -------------------------

        z1 = (x1 * w11) + (x2 * w12) + (x3 * w13) + b1

        if z1 > 0:
            h1 = z1
        else:
            h1 = 0

        z2 = (x1 * w21) + (x2 * w22) + (x3 * w23) + b2

        if z2 > 0:
            h2 = z2
        else:
            h2 = 0

        z3 = (h1 * v1) + (h2 * v2) + b3

        probability = 1 / (1 + math.exp(-z3))

        # -------------------------
        # LOSS
        # -------------------------

        epsilon = 0.0000001

        if probability < epsilon:
            p = epsilon
        elif probability > 1 - epsilon:
            p = 1 - epsilon
        else:
            p = probability

        loss = -(
            y * math.log(p)
            +
            (1 - y) * math.log(1 - p)
        )

        total_loss = total_loss + loss

        # -------------------------
        # BACKPROPAGATION
        # -------------------------

        dz3 = probability - y

        dv1 = dz3 * h1
        dv2 = dz3 * h2
        db3 = dz3

        dh1 = dz3 * v1
        dh2 = dz3 * v2

        if z1 > 0:
            dz1 = dh1
        else:
            dz1 = 0

        if z2 > 0:
            dz2 = dh2
        else:
            dz2 = 0

        dw11 = dz1 * x1
        dw12 = dz1 * x2
        dw13 = dz1 * x3
        db1 = dz1

        dw21 = dz2 * x1
        dw22 = dz2 * x2
        dw23 = dz2 * x3
        db2 = dz2

        # -------------------------
        # UPDATE WEIGHTS
        # -------------------------

        w11 = w11 - learning_rate * dw11
        w12 = w12 - learning_rate * dw12
        w13 = w13 - learning_rate * dw13
        b1 = b1 - learning_rate * db1

        w21 = w21 - learning_rate * dw21
        w22 = w22 - learning_rate * dw22
        w23 = w23 - learning_rate * dw23
        b2 = b2 - learning_rate * db2

        v1 = v1 - learning_rate * dv1
        v2 = v2 - learning_rate * dv2
        b3 = b3 - learning_rate * db3

    # Show loss after every 100 epochs
    if (epoch + 1) % 100 == 0:
        average_loss = total_loss / len(rows)
        print(
            "Epoch:",
            epoch + 1,
            "Average Loss:",
            round(average_loss, 4)
        )


# ------------------------------------------------------------
# STEP 4: FINAL TESTING
# ------------------------------------------------------------

correct = 0
total_loss = 0

print()
print("=" * 55)
print("FINAL TEST RESULTS")
print("=" * 55)

for row in rows:

    x1 = float(row["temperature_c"]) / 100
    x2 = float(row["humidity_percent"]) / 100
    x3 = (float(row["pressure_hpa"]) - 1000) / 20

    y = int(row["rain_next_hour"])

    # Forward pass

    z1 = (x1 * w11) + (x2 * w12) + (x3 * w13) + b1

    if z1 > 0:
        h1 = z1
    else:
        h1 = 0

    z2 = (x1 * w21) + (x2 * w22) + (x3 * w23) + b2

    if z2 > 0:
        h2 = z2
    else:
        h2 = 0

    z3 = (h1 * v1) + (h2 * v2) + b3

    probability = 1 / (1 + math.exp(-z3))

    if probability >= 0.5:
        predicted_class = 1
    else:
        predicted_class = 0

    if predicted_class == y:
        correct = correct + 1

    print(
        row["timestamp"],
        "| Actual:",
        y,
        "| Predicted:",
        predicted_class,
        "| Probability:",
        round(probability, 4)
    )


# ------------------------------------------------------------
# STEP 5: ACCURACY
# ------------------------------------------------------------

accuracy = correct / len(rows)

print()
print("=" * 55)
print("FINAL RESULTS")
print("=" * 55)
print("Correct predictions =", correct, "out of", len(rows))
print("Accuracy =", round(accuracy, 4))
print("Accuracy percentage =", round(accuracy * 100, 2), "%")