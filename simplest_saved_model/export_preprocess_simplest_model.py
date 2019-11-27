#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import marshal
import types
import os
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (
    signature_constants, signature_def_utils, tag_constants, utils)
from tensorflow.python.util import compat


def preprocess(input_dict):
  # Import dependencies
  import jieba

  preprocess_input_dict = {"keys": []}

  # Process data
  if "keys" in input_dict:
    for item in input_dict["keys"]:
      seg_list = jieba.cut(item, cut_all=True)
      token_number = sum(1 for i in seg_list)
      preprocess_input_dict["keys"].append(token_number)

  # Return data
  # print("Input after preprocess: {}".format(preprocess_input_dict))
  return preprocess_input_dict


def postprocess(output_dict):

  postprocess_output_dict = {"keys": []}

  # Process data
  if "keys" in output_dict:
    for item in output_dict["keys"]:
      item_string = "Token number: {}".format(item)
      postprocess_output_dict["keys"].append(item_string)

  # Return data
  # print("Output after postprocess: {}".format(postprocess_output_dict))
  return postprocess_output_dict


def test_preprocess():
  input = {"keys": ["你好世界", "机器学习模型服务预处理"]}
  new_input = preprocess(input)
  print(new_input)

  preprocess_function_string = marshal.dumps(preprocess.func_code)
  loaded_preprocess_function = marshal.loads(preprocess_function_string)
  load_preprocess_func = types.FunctionType(loaded_preprocess_function,
                                            globals(), "preprocess_function")
  load_preprocess_func({"keys": ["新的你好世界", "新的机器学习模型服务预处理"]})
  string_from_collection = tf.get_collection("preprocess_function")
  print(string_from_collection)


def test_postprocess():
  output = {'keys': [2, 5]}
  new_output = postprocess(output)
  print(new_output)


def main():
  # test_preprocess()
  # test_postprocess()

  preprocess_function_string = marshal.dumps(preprocess.func_code)
  tf.add_to_collection("preprocess_function", preprocess_function_string)
  postprocess_function_string = marshal.dumps(postprocess.func_code)
  tf.add_to_collection("postprocess_function", postprocess_function_string)

  model_path = "preprocess_model"
  model_version = 1

  keys_placeholder = tf.placeholder(tf.int32, shape=[None], name="keys")
  keys_identity = tf.identity(keys_placeholder, name="inference_keys")

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  model_signature = signature_def_utils.build_signature_def(
      inputs={
          "keys": utils.build_tensor_info(keys_placeholder),
      },
      outputs={
          "keys": utils.build_tensor_info(keys_identity),
      },
      method_name=signature_constants.PREDICT_METHOD_NAME)

  export_path = os.path.join(
      compat.as_bytes(model_path), compat.as_bytes(str(model_version)))

  builder = saved_model_builder.SavedModelBuilder(export_path)
  builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      clear_devices=True,
      signature_def_map={
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          model_signature,
      })

  builder.save()


if __name__ == "__main__":
  main()
