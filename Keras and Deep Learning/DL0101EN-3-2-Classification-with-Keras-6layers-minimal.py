import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.utils import to_categorical

import matplotlib.pyplot as plt

# import the data
from keras.datasets import mnist

# read the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# show one example image (optional)
plt.imshow(X_train[0], cmap="gray")
plt.axis("off")
plt.show()

# flatten images into one-dimensional vector
num_pixels = X_train.shape[1] * X_train.shape[2]  # 28*28 = 784

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype("float32")
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype("float32")

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]
print(num_classes)

# define classification model (6 Dense layers: 5 hidden + 1 output)
def classification_model():
    model = Sequential(name="classification_mlp_6_dense")

    model.add(Input(shape=(num_pixels,)))
    model.add(Dense(num_pixels, activation="relu"))      # hidden 1
    model.add(Dense(100, activation="relu"))             # hidden 2
    model.add(Dense(100, activation="relu"))             # hidden 3
    model.add(Dense(100, activation="relu"))             # hidden 4
    model.add(Dense(100, activation="relu"))             # hidden 5
    model.add(Dense(num_classes, activation="softmax"))  # output

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# build the model
model = classification_model()

# fit the model
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=10,
          verbose=2)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {}% \n Error: {}".format(scores[1], 1 - scores[1]))

# save model
model.save("classification_model.keras")

# load model back (optional)
pretrained_model = keras.saving.load_model("classification_model.keras")
