#!/usr/bin/env python

import os
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
loss = tf.square(Y - X * w - b)
global_step = tf.Variable(0, name="global_step", trainable=False)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss, global_step=global_step)

# Init checkpoint
saver = tf.train.Saver()
checkpoint_path = "./checkpoint"
checkpoint_file_path = checkpoint_path + "/checkpoint.ckpt"
latest_checkpoint_file_path = tf.train.latest_checkpoint(checkpoint_path)

# Create session to run
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  # Restore checkpoint
  if os.path.isdir(checkpoint_path):
    print("Restore checkpoint from {}".format(checkpoint_path))
    saver.restore(sess, latest_checkpoint_file_path)

  epoch = 1
  for i in range(10):
    for (x, y) in zip(train_X, train_Y):
      _, w_value, b_value, global_step_value= sess.run(
          [train_op, w, b, global_step], feed_dict={X: x, Y: y})

    print("Epoch: {}, w: {}, b: {}".format(epoch, w_value, b_value))
    epoch += 1

  saver.save(sess, checkpoint_file_path, global_step=global_step_value)
  print("Save checkpoint in {}".format(checkpoint_file_path))



