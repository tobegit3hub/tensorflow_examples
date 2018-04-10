#!/usr/bin/env python

import numpy as np
import tensorflow as tf

# Prepare train data
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10

# Define the model
X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")
import ipdb
ipdb.set_trace()
temp = tf.add(w, w)
loss = tf.square(Y - X * w - b)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# Create session to run
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  profiler = tf.profiler.Profiler(sess.graph)
  meta = tf.RunMetadata()

  epoch = 1
  for i in range(10):
    for (x, y) in zip(train_X, train_Y):
      _, w_value, b_value = sess.run(
          [train_op, w, b],
          feed_dict={X: x,
                     Y: y},
          options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
          run_metadata=meta)
      profiler.add_step(i, meta)

    print("Epoch: {}, w: {}, b: {}".format(epoch, w_value, b_value))
    epoch += 1

  opts = tf.profiler.ProfileOptionBuilder.time_and_memory()

  profiler.profile_operations(options=opts)
