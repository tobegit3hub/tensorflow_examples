#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout
from keras.optimizers import SGD


def main():
  # Define hyper-parameters
  OUTPUT_CLASS = 10
  EPOCH_NUMBER = 10
  BATCH_SIZE = 128
  VERBOSE = 1
  OPTIMIZER = SGD()
  VALIDATION_SPLIT = 0.2
  LOSS_TYPE = "categorical_crossentropy"

  # Get training dataset
  # (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data(
  #     "/Users/tobe/.keras/datasets/mnist.npz")
  (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
  X_train = X_train.reshape(60000, 784)
  X_test = X_test.reshape(10000, 784)
  X_train = X_train.astype("float32")
  X_test = X_test.astype("float32")
  X_train = X_train / 255
  X_test = X_test / 255
  Y_train = keras.utils.np_utils.to_categorical(Y_train, OUTPUT_CLASS)
  Y_test = keras.utils.np_utils.to_categorical(Y_test, OUTPUT_CLASS)

  # Build model
  # model = build_model()
  model = build_dnn_model()
  model_json = model.to_json()
  print("Model json: {}".format(model_json))
  # model.save("./mnist_dnn_model.h5")
  # keras.models.model_from_json()
  # keras.models.load_model()
  # keras.callbacks.TensorBoard(log_dir="./tensorboard", histogram_freq=0, write_grads=True, write_images=False)
  model.summary()
  model.compile(loss=LOSS_TYPE, optimizer=OPTIMIZER, metrics=["accuracy"])

  # Train model
  model.fit(
      X_train,
      Y_train,
      batch_size=BATCH_SIZE,
      epochs=EPOCH_NUMBER,
      validation_split=VALIDATION_SPLIT,
      verbose=VERBOSE)

  # Make prediction
  metrics = model.evaluate(X_test, Y_test, verbose=VERBOSE)
  test_loss = metrics[0]
  test_accuracy = metrics[1]
  print("Accuracy: {}".format(test_accuracy))

  return test_accuracy


def build_model():
  model = keras.models.Sequential()

  model.add(Dense(10, input_shape=(784, )))
  model.add(Activation("softmax"))

  return model


def build_dnn_model():
  model = keras.models.Sequential()

  model.add(Dense(128, input_shape=(784, )))
  model.add(Activation("relu"))
  model.add(Dropout(0.5))

  model.add(Dense(64))
  model.add(Activation("relu"))
  model.add(Dropout(0.5))

  model.add(Dense(32))
  model.add(Activation("relu"))
  model.add(Dropout(0.5))

  model.add(Dense(10))
  model.add(Activation("softmax"))

  return model


if __name__ == "__main__":
  main()
