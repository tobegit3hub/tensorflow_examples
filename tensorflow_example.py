#!/usr/bin/env python

import numpy as np
import tensorflow as tf

X_train = np.linspace(-1, 1, 100)
Y_train = 2 * X_train + 10 + np.random.randn(*X_train.shape) * 0.33
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")

loss = tf.square(Y - X * w - b)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  for i in range(10):
    for (x, y) in zip(X_train, Y_train):
      _, loss_value = sess.run([train_op, loss], feed_dict={X: x, Y: y})
      print("Loss: {}".format(loss_value))