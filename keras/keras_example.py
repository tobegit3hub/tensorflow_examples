#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
import numpy as np

def main():
  X_train = np.random.random((1000, 784))
  Y_train = np.random.randint(2, size=(1000, 1))

  model = keras.models.Sequential()
  model.add(keras.layers.core.Dense(1, input_shape=(784, ), activation="softmax"))
  optimizer = keras.optimizers.SGD()
  model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
  model.fit(X_train, Y_train, batch_size=9, epochs=10, validation_split=0.2, verbose=1)

if __name__ == "__main__":
  main()
