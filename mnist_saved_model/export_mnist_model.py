#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def inference(input):
  weights = tf.get_variable(
      "weights", [784, 10], initializer=tf.random_normal_initializer())
  bias = tf.get_variable(
      "bias", [10], initializer=tf.random_normal_initializer())
  logits = tf.matmul(input, weights) + bias

  return logits


def main():
  mnist = input_data.read_data_sets("./input_data")

  x = tf.placeholder(tf.float32, [None, 784])
  logits = inference(x)
  y_ = tf.placeholder(tf.int64, [None])
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      labels=y_, logits=logits)
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  init_op = tf.global_variables_initializer()

  # Define op for model signature
  tf.get_variable_scope().reuse_variables()

  model_base64_placeholder = tf.placeholder(
      shape=[None], dtype=tf.string, name="model_input_b64_images")
  model_base64_string = tf.decode_base64(model_base64_placeholder)
  model_base64_input = tf.map_fn(lambda x: tf.image.resize_images(tf.image.decode_jpeg(x, channels=1), [28, 28]), model_base64_string, dtype=tf.float32)
  model_base64_reshape_input = tf.reshape(model_base64_input, [-1, 28 * 28])
  model_logits = inference(model_base64_reshape_input)
  model_predict_softmax = tf.nn.softmax(model_logits)
  model_predict = tf.argmax(model_predict_softmax, 1)

  with tf.Session() as sess:

    sess.run(init_op)

    for i in range(938):
      batch_xs, batch_ys = mnist.train.next_batch(64)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Export image model
    export_dir = "./model/1"
    print("Try to export the model in {}".format(export_dir))
    tf.saved_model.simple_save(
        sess,
        export_dir,
        inputs={"images": model_base64_placeholder},
        outputs={
            "predict": model_predict,
            "probability": model_predict_softmax
        })


if __name__ == "__main__":
  main()
