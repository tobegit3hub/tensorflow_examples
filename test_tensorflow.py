#!/usr/bin/env python

import tensorflow as tf
import numpy as np

train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10

X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")

cost_op = tf.square(Y - tf.mul(X, w) - b)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost_op)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(10):
        for (x, y) in zip(train_X, train_Y):
            sess.run(train_op, feed_dict={X: x, Y: y})

    if 1 < sess.run(w) < 3 and 9 < sess.run(b) < 11:
        print("Success")
    else:
        print("Fail")
