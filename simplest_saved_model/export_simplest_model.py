#!/usr/bin/env python

import tensorflow as tf

export_dir = "./model/1"
input_keys_placeholder = tf.placeholder(
    tf.int32, shape=[None, 1], name="input_keys")
output_keys = tf.identity(input_keys_placeholder, name="output_keys")

session = tf.Session()
tf.saved_model.simple_save(
    session,
    export_dir,
    inputs={"keys": input_keys_placeholder},
    outputs={"keys": output_keys})
