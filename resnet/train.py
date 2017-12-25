#!/usr/bin/env python

import cPickle

import numpy as np
import tensorflow as tf

import model


def unpickle(file):
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  return dict


def one_hot_vec(label):
  vec = np.zeros(10)
  vec[label] = 1
  return vec


def load_data():
  x_all = []
  y_all = []
  for i in range(5):
    d = unpickle("cifar-10-batches-py/data_batch_" + str(i + 1))
    x_ = d['data']
    y_ = d['labels']
    x_all.append(x_)
    y_all.append(y_)

  d = unpickle('cifar-10-batches-py/test_batch')
  x_all.append(d['data'])
  y_all.append(d['labels'])

  x = np.concatenate(x_all) / np.float32(255)
  y = np.concatenate(y_all)
  x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
  x = x.reshape((x.shape[0], 32, 32, 3))

  pixel_mean = np.mean(x[0:50000], axis=0)
  x -= pixel_mean

  y = map(one_hot_vec, y)
  X_train = x[0:50000, :, :, :]
  Y_train = y[0:50000]
  X_test = x[50000:, :, :, :]
  Y_test = y[50000:]

  return (X_train, Y_train, X_test, Y_test)


def main():
  # Define hyper-parameter
  learning_rate = 0.01
  batch_size = 12
  epoch_number = 1
  steps_to_validate = 12
  resnet_layer_number = 32  # 20

  # Load training dataset
  X_train, Y_train, X_test, Y_test = load_data()

  # Define the model
  X = tf.placeholder("float", [None, 32, 32, 3])
  Y = tf.placeholder("float", [None, 10])
  net = model.resnet(X, resnet_layer_number)
  cross_entropy = -tf.reduce_sum(Y * tf.log(net))
  opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
  train_op = opt.minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

  # Define other tools
  saver = tf.train.Saver()
  checkpoint = tf.train.latest_checkpoint("./checkpoint/")
  init_op = tf.initialize_all_variables()

  # Start the session
  with tf.Session() as sess:
    sess.run(init_op)

    # Restore from checkpoint
    if checkpoint:
      print("Restore checkpoint from: {}".format(checkpoint))
      #saver.restore(sess, checkpoint)

    # Start training
    for epoch_index in range(epoch_number):
      for i in range(0, 50000, batch_size):
        feed_dict = {
            X: X_train[i:i + batch_size],
            Y: Y_train[i:i + batch_size]
        }
        sess.run([train_op], feed_dict=feed_dict)

        if i % steps_to_validate == 0:
          saver.save(sess, './checkpoint/', global_step=i)

          validate_start_index = 0
          validate_end_index = validate_start_index + batch_size
          valiate_accuracy_value = sess.run(
              [accuracy],
              feed_dict={
                  X: X_test[validate_start_index:validate_end_index],
                  Y: Y_test[validate_start_index:validate_end_index]
              })
          print("Epoch: {}, image id: {}, validate accuracy: {}".format(
              epoch_index, i, valiate_accuracy_value))


if __name__ == "__main__":
  main()
