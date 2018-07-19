#!/usr/bin/env python

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils


def input_fn():
  features = {'a': tf.constant([["1"], ["2"]]), 'b': tf.constant([[3], [4]])}
  labels = tf.constant([0, 1])
  return features, labels


#feature_a = tf.contrib.layers.sparse_column_with_integerized_feature("a", bucket_size=10)
#feature_b = tf.contrib.layers.sparse_column_with_integerized_feature("b", bucket_size=10)
#feature_c = tf.contrib.layers.crossed_column([feature_a, feature_b], hash_bucket_size=100)
#feature_columns = [feature_a, feature_b, feature_c]

feature_a = tf.contrib.layers.sparse_column_with_hash_bucket(
    "a", hash_bucket_size=1000)
feature_b = tf.contrib.layers.real_valued_column("b")
feature_columns = [feature_a, feature_b]

model = tf.contrib.learn.LinearClassifier(feature_columns=feature_columns)
model.fit(input_fn=input_fn, steps=10)

feature_spec = tf.contrib.layers.create_feature_spec_for_parsing(
    feature_columns)
serving_input_fn = input_fn_utils.build_parsing_serving_input_fn(feature_spec)

savedmodel_path = "./savedmodel"
model.export_savedmodel(savedmodel_path, serving_input_fn)
