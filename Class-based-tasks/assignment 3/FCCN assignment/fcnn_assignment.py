import math
def weighted_sum(x1, x2, x3, w1, w2, w3, b):
    z = w1*x1 + w2*x2 + w3*x3 + b
    return z


x1 = 0.8
x2 = 0.6
x3 = 0.7

w1 = 0.5
w2 = 0.2
w3 = 0.4

b = -0.1

z = weighted_sum(x1, x2, x3, w1, w2, w3, b)

print("Weighted Sum =", z)

def relu(z):
    return max(0, z)
print(relu(-0.8))
print(relu(0))
print(relu(0.36))
print(relu(1.4))
def hidden_neuron_1(x1, x2, x3):
    z1 = 0.5*x1 + 0.2*x2 + 0.4*x3 - 0.1
    h1 = relu(z1)
    return z1, h1
z1, h1 = hidden_neuron_1(0.8, 0.6, 0.7)
print("z1 =", z1)
print("h1 =", h1)
def hidden_neuron_2(x1, x2, x3):
    z2 = -0.3*x1 + 0.8*x2 + 0.1*x3 + 0.05
    h2 = relu(z2)
    return z2, h2
z2, h2 = hidden_neuron_2(0.8, 0.6, 0.7)

print("z2 =", z2)
print("h2 =", h2)
z3 = 1.0*h1 + 0.7*h2 - 0.2

print("z3 =", z3)

def sigmoid(z):
    return 1 / (1 + math.exp(-z))
p = sigmoid(z3)

print("Probability =", p)
def bce_loss(y, p):
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))
y = 1

loss = bce_loss(y, p)

print("Loss when y=1:", loss)
y = 1
loss = bce_loss(y, p)
print("Loss when y=1:", loss)

y = 0

loss = bce_loss(y, p)

print("Loss when y=0:", loss)








