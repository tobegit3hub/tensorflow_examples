#!/usr/bin/env python

import numpy as np
import tensorflow as tf

# Prepare train data
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10

# Define the model
X = tf.placeholder(tf.float32, shape=[2])
Y = tf.placeholder(tf.float32, shape=[2])
w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")
predict = X * w + b

loss = tf.square(Y - predict)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# Create session to run
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  epoch = 1
  for i in range(10):
    for (x, y) in zip(train_X, train_Y):
      _, w_value, b_value = sess.run([train_op, w, b], feed_dict={X: [x], Y: [y]})
    print("Epoch: {}, w: {}, b: {}".format(epoch, w_value, b_value))
    epoch += 1

  export_dir = "./model/1/"
  print("Try to export the model in {}".format(export_dir))
  tf.saved_model.simple_save(sess, export_dir, inputs={"x": X}, outputs={"y": predict})

