"""
Convolutional Neural Networks with Keras 

This script loads the MNIST dataset, preprocesses it, and trains/evaluates:
1. A baseline CNN
2. A deeper CNN
3. The deeper CNN with a larger batch size
4. The deeper CNN with more epochs
"""

import os
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical


# -----------------------------
# Data loading and preprocessing
# -----------------------------
def load_and_prepare_data():
    """Load MNIST and prepare it for CNN training."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape to: (samples, height, width, channels)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32")
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32")

    # Normalize pixel values
    x_train /= 255.0
    x_test /= 255.0

    # One-hot encode labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    num_classes = y_test.shape[1]
    return x_train, y_train, x_test, y_test, num_classes


# -----------------------------
# Model definitions
# -----------------------------
def build_baseline_cnn(num_classes: int) -> Sequential:
    """Build the baseline CNN model."""
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(16, (5, 5), strides=(1, 1), activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(100, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_deeper_cnn(num_classes: int) -> Sequential:
    """Build the deeper CNN model."""
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(16, (5, 5), activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(8, (2, 2), activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(100, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -----------------------------
# Training and evaluation helper
# -----------------------------
def train_and_evaluate(
    model: Sequential,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs: int,
    batch_size: int,
    label: str,
):
    """Train and evaluate a model, then print the results."""
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    error_rate = 100 - accuracy * 100

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Error: {error_rate:.2f}%")

    return {"loss": loss, "accuracy": accuracy, "error_rate": error_rate}


# -----------------------------
# Main program
# -----------------------------
def main():
    x_train, y_train, x_test, y_test, num_classes = load_and_prepare_data()

    # 1. Baseline model
    baseline_model = build_baseline_cnn(num_classes)
    train_and_evaluate(
        baseline_model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=10,
        batch_size=200,
        label="Baseline CNN (10 epochs, batch_size=200)",
    )

    # 2. Deeper model
    deeper_model = build_deeper_cnn(num_classes)
    train_and_evaluate(
        deeper_model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=10,
        batch_size=200,
        label="Deeper CNN (10 epochs, batch_size=200)",
    )

    # 3. Deeper model with larger batch size
    larger_batch_model = build_deeper_cnn(num_classes)
    train_and_evaluate(
        larger_batch_model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=10,
        batch_size=1024,
        label="Deeper CNN (10 epochs, batch_size=1024)",
    )

    # 4. Deeper model with more epochs
    more_epochs_model = build_deeper_cnn(num_classes)
    train_and_evaluate(
        more_epochs_model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=25,
        batch_size=1024,
        label="Deeper CNN (25 epochs, batch_size=1024)",
    )


if __name__ == "__main__":
    main()
