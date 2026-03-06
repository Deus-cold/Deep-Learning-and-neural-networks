

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# ----------------------
# Load MNIST
# ----------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Optional: show the first image
plt.imshow(X_train[0], cmap='gray')
plt.axis('off')
plt.show()

# ----------------------
# Prepare data
# ----------------------
num_pixels = X_train.shape[1] * X_train.shape[2]  # 28*28 = 784

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype("float32") / 255.0
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype("float32") / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]  # 10


# ----------------------
# Build model (6 Dense layers: 5 hidden + 1 output)
# ----------------------
def classification_model():
    model = Sequential()
    model.add(Input(shape=(num_pixels,)))
    model.add(Dense(num_pixels, activation="relu"))   # hidden 1
    model.add(Dense(100, activation="relu"))          # hidden 2
    model.add(Dense(100, activation="relu"))          # hidden 3
    model.add(Dense(100, activation="relu"))          # hidden 4
    model.add(Dense(100, activation="relu"))          # hidden 5
    model.add(Dense(num_classes, activation="softmax"))  # output

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ----------------------
# Train + Evaluate
# ----------------------
model = classification_model()

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=10,
          verbose=2)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc*100:.2f}% | Test loss: {loss:.4f}")

# Save model
model.save("classification_model.keras")
print("Saved model to classification_model.keras")
