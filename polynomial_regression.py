#!/usr/bin/env python

import tensorflow as tf
import numpy as np

# Prepare train data
train_X = np.linspace(-1, 1, 100)
train_Y = 10 + 1 * train_X + 2 * np.power(train_X, 2) + 3 * np.power(
    train_X, 3) + np.random.randn(*train_X.shape) * 0.33

# Define the model
X = tf.placeholder("float")
Y = tf.placeholder("float")
w1 = tf.Variable(0.0, name="weight1")
w2 = tf.Variable(0.0, name="weight2")
w3 = tf.Variable(0.0, name="weight3")
b = tf.Variable(0.0, name="bias")
loss = tf.square(Y - b - tf.mul(w1, X) - tf.mul(w2, tf.pow(X, 2)) - tf.mul(
    w3, tf.pow(X, 3)))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
train_epoch_number = 100

# Create session to run
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    epoch = 1
    for i in range(train_epoch_number):
        for (x, y) in zip(train_X, train_Y):
            _, b_value, w1_value, w2_value, w3_value = sess.run(
                [train_op, b, w1, w2, w3],
                feed_dict={X: x,
                           Y: y})
        print("Epoch: {}, b: {}, w1: {}, w2: {}, w3: {}".format(
            epoch, b_value, w1_value, w2_value, w3_value))
        epoch += 1
