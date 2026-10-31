
import tensorflow as tf
from keras import layers, models

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)

# Normalize pixel values from 0-255 to 0-1
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Create the CNN model
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),

    layers.Conv2D(
        32, (3, 3), activation="relu"
    ),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        64, (3, 3), activation="relu"
    ),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display the model structure
model.summary()

# Train the CNN
model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# Evaluate on unseen test images
test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=2
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")

# Predict the first test image
predictions = model.predict(X_test[:1], verbose=0)
print("Predicted class:", predictions.argmax())
print("Actual class:", y_test[0][0])
