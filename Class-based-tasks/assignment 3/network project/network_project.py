import csv
import math

# ------------------------------------------------------------
# STEP 1: LOAD NETWORK DATA
# ------------------------------------------------------------

rows = []

with open("network_data.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        rows.append(row)

print("Total network records loaded:", len(rows))
print()


# ------------------------------------------------------------
# STEP 2: FCNN WEIGHTS
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


# ------------------------------------------------------------
# STEP 3: FORWARD PASS + LOSS + ACCURACY
# ------------------------------------------------------------

total_loss = 0
correct = 0
epsilon = 0.0000001

for row in rows:

    # Input values
    x1 = float(row["ping_ms"]) / 100
    x2 = float(row["jitter_ms"]) / 40
    x3 = float(row["packet_loss_percent"]) / 10

    # Actual answer
    y = int(row["poor_connection"])

    # Hidden neuron 1
    z1 = (x1 * w11) + (x2 * w12) + (x3 * w13) + b1

    # ReLU
    if z1 > 0:
        h1 = z1
    else:
        h1 = 0

    # Hidden neuron 2
    z2 = (x1 * w21) + (x2 * w22) + (x3 * w23) + b2

    # ReLU
    if z2 > 0:
        h2 = z2
    else:
        h2 = 0

    # Output neuron
    z3 = (h1 * v1) + (h2 * v2) + b3

    # Sigmoid
    probability = 1 / (1 + math.exp(-z3))

    # Prediction
    if probability >= 0.5:
        predicted_class = 1
    else:
        predicted_class = 0

    # Loss
    if probability < epsilon:
        probability_for_loss = epsilon
    elif probability > 1 - epsilon:
        probability_for_loss = 1 - epsilon
    else:
        probability_for_loss = probability

    loss = -(y * math.log(probability_for_loss) +
             (1 - y) * math.log(1 - probability_for_loss))

    total_loss = total_loss + loss

    # Accuracy
    if predicted_class == y:
        correct = correct + 1

    print(
        "Actual:", y,
        "Predicted:", predicted_class,
        "Probability:", round(probability, 4)
    )


# ------------------------------------------------------------
# STEP 4: FINAL RESULTS
# ------------------------------------------------------------

average_loss = total_loss / len(rows)
accuracy = correct / len(rows)

print()
print("Average Loss:", round(average_loss, 4))
print("Correct Predictions:", correct, "/", len(rows))

print("Accuracy:", round(accuracy * 100, 2), "%")
