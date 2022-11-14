# coding: utf-8

import tensorflow as tf
#from tensorflow.contrib.layers import *
import numpy as np

def _new_embedding(name, sparse_id_feature_column, config, trainable=True):
  embedding_fc = None
  if name == 'embedding':
    embedding_fc = tf.contrib.layers.embedding_column(
      sparse_id_column=sparse_id_feature_column,
      dimension=config['embedding_dimension'],
      combiner=None,
      initializer=tf.contrib.layers.xavier_initializer(),
      trainable=trainable
    )
  elif name == 'shared_embedding':
    embedding_fc = tf.contrib.layers.shared_embedding_columns(
      sparse_id_columns=[sparse_id_feature_column],
      dimension=config['embedding_dimension'],
      combiner=None,
      shared_embedding_name=config['shared_embedding_name'],
      initializer=tf.contrib.layers.xavier_initializer(),
      trainable=trainable
    )[0]

  else:
    print "embedding type error!"

  return embedding_fc

def _new_feature_columns(name, config, trainable=True):
  fc = None
  embedding = None
  if name == 'sparse_column_with_integerized_feature':
    sparse_id_fc = tf.contrib.layers.sparse_column_with_integerized_feature(
      column_name=config['feature_column_name'],
      bucket_size=config['num_buckets'],
      dtype=config['dtype']
    )
    if config.has_key('embedding'):
      fc = _new_embedding(config['embedding'], sparse_id_fc, config, trainable)
    else:
      fc = sparse_id_fc

  elif name == 'sparse_column_with_hash_bucket':
    hash_bucket_fc = tf.contrib.layers.sparse_column_with_hash_bucket(
      column_name=config['feature_column_name'],
      hash_bucket_size=config['hash_bucket_size'],
      dtype=config['dtype']
    )
    if config.has_key('embedding'):
      fc = _new_embedding(config['embedding'], hash_bucket_fc, config, trainable)
    else:
      fc = hash_bucket_fc
  elif name == 'categorical_column_with_vocabulary_list':
    spare_id_fc = tf.feature_column.sparse_column_with_keys(
      key=config['feature_column_name'],
      keys=config['keys'],
      default_value=config['default_value'],
      dtype=config['dtype']
    )
    if config.has_key('embedding'):
      fc = _new_embedding(config['embedding'], spare_id_fc, config, trainable)
    else:
      fc = spare_id_fc
  elif name == 'bucketized_column':
    numeric_column = tf.contrib.layers.real_valued_column(
      column_name=config['feature_column_name']
    )
    bucketized_column = tf.contrib.layers.bucketized_column(numeric_column, config['boundaries'])
    if config.has_key('embedding'):
      fc = _new_embedding(config['embedding'], bucketized_column, config, trainable)
    else:
      fc = bucketized_column
  else:
    print "error in feature_column type"

  if fc is None:
    print "Faild to get fc!"
  return fc


def new_input_layer(input_tensor, feature_column_config, trainable=True):
  feature_column_list = []
  for e in feature_column_config:
    if e.has_key('feature_column_type'):
      fc = _new_feature_columns(e['feature_column_type'], e, trainable)
      feature_column_list.append(fc)
  print input_tensor
  print feature_column_list
  return tf.contrib.layers.input_from_feature_columns(input_tensor, feature_column_list)

# def get_input_config(config, list):
#   res_config = {}
#   for e in list:
#     res_config[e] = config[e]
#   return res_config