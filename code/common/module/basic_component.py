# coding=utf-8

import tensorflow as tf
import math
import numpy as np
import random
import sys


def _add_conv_single_layer(input, seq_length, embed_size, filter_num, k, name='', use_bias=True, trainable=True):

  with tf.variable_scope('conv_single_' + str(k) + name):
    window_size = k
    conv_W = tf.get_variable("convW_width",
                             shape=[window_size, embed_size, 1, filter_num],
                             trainable=trainable,
                             initializer=tf.truncated_normal_initializer(stddev=0.01))
    x_query_pad_width2 = tf.pad(input,
                                [[0, 0], [window_size - 1, 0], [0, 0], [0, 0]],
                                "CONSTANT")
    y = tf.nn.conv2d(x_query_pad_width2, conv_W, strides=[1, 1, 1, 1], padding='VALID')
    y = tf.reshape(y, [-1, seq_length, filter_num, 1])
    if use_bias:
      conv_B = tf.get_variable('convB_width',
                               shape=[filter_num, 1],
                               trainable=trainable,
                               initializer=tf.constant_initializer(0))
      return y + conv_B
    else:
      return y






def _add_conv_shared_layer(w2v, embed_size, filter_num, batch_size, k, name='', trainable=True):

  with tf.variable_scope("conv_layer"+name, reuse=tf.AUTO_REUSE):
    # # width=1
    # filter_width = 1
    # conv_W = tf.get_variable("convW_width1",
    #                                        shape=[filter_width, embed_size, 1, filter_num],
    #                                        initializer=tf.truncated_normal_initializer(stddev=0.01))
    # x_query_pad_width1 = tf.pad(w2v,
    #                             [[0, 0], [filter_width - 1, 0], [0, 0], [0, 0]],
    #                             "CONSTANT")
    # y = tf.nn.conv2d(x_query_pad_width1, conv_W, strides=[1, 1, 1, 1], padding='VALID')
    # # pooling values, indices = tf.nn.top_k(tf.transpose(y2, [0, 2, 1]), int(config.k))
    # hidden, _ = tf.nn.top_k(tf.transpose(y, [0, 2, 3, 1]), int(k))
    # query_width1 = tf.reshape(hidden, [batch_size, -1])

    # width=2
    filter_width = 2
    conv_W = tf.get_variable("convW_width2",
                             shape=[filter_width, embed_size, 1, filter_num],
                             trainable=trainable,
                             initializer=tf.truncated_normal_initializer(stddev=0.01))
    x_query_pad_width2 = tf.pad(w2v,
                                [[0, 0], [filter_width - 1, 0], [0, 0], [0, 0]],
                                "CONSTANT")
    y = tf.nn.conv2d(x_query_pad_width2, conv_W, strides=[1, 1, 1, 1], padding='VALID')
    # pooling values, indices = tf.nn.top_k(tf.transpose(y2, [0, 2, 1]), int(config.k))
    hidden, _ = tf.nn.top_k(tf.transpose(y, [0, 2, 3, 1]), int(k))
    query_width2 = tf.reshape(hidden, [batch_size, filter_num])

    # width = 3
    filter_width = 3
    conv_W = tf.get_variable("convW_width3",
                             shape=[filter_width, embed_size, 1, filter_num],
                             trainable=trainable,
                             initializer=tf.truncated_normal_initializer(stddev=0.01))
    x_query_pad_width3 = tf.pad(w2v,
                                [[0, 0], [filter_width - 1, 0], [0, 0], [0, 0]],
                                "CONSTANT")
    y = tf.nn.conv2d(x_query_pad_width3, conv_W, strides=[1, 1, 1, 1], padding='VALID')
    # pooling values, indices = tf.nn.top_k(tf.transpose(y2, [0, 2, 1]), int(config.k))
    hidden, indices = tf.nn.top_k(tf.transpose(y, [0, 2, 3, 1]), int(k))
    query_width3 = tf.reshape(hidden, [batch_size, filter_num])

    # width = 4
    filter_width = 4
    conv_W = tf.get_variable("convW_width4",
                             shape=[filter_width, embed_size, 1, filter_num],
                             trainable=trainable,
                             initializer=tf.truncated_normal_initializer(stddev=0.01))
    x_query_pad_width4 = tf.pad(w2v,
                                [[0, 0], [filter_width - 1, 0], [0, 0], [0, 0]],
                                "CONSTANT")

    y = tf.nn.conv2d(x_query_pad_width4, conv_W, strides=[1, 1, 1, 1], padding='VALID')
    hidden, indices = tf.nn.top_k(tf.transpose(y, [0, 2, 3, 1]), int(k))
    query_width4 = tf.reshape(hidden, [batch_size, filter_num])


    query_concat  = tf.concat([query_width2, query_width3, query_width4], 1)
  return query_concat

def _batch_norm(x, n_out, phase_train, scope='bn', trainable=True):
  """
  Batch normalization on convolutional maps.
  Args:
      x:           Tensor, 4D BHWD input maps
      n_out:       integer, depth of input maps
      phase_train: boolean tf.Varialbe, true indicates training phase
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  with tf.variable_scope(scope):
    beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0),
                           trainable=trainable)
    gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.constant_initializer(1),
                            trainable=trainable)
    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    # if phase_train == True:
    #   mean, var = mean_var_with_update()
    # else:
    #   mean, var = ema.average(batch_mean), ema.average(batch_var)
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
  return normed

# based on shape [batch, dim]
def _add_fc_layer_shared_by_name(inputs, in_size, out_size, name, trainable_tensor=None, activation_function=None, regularizer = None,
                                 batch_normalize_name = None, user_bias=True, trainable=True):

  with tf.variable_scope("fc_layer", reuse=tf.AUTO_REUSE):
    wlimit = np.sqrt(6.0 / (in_size + out_size))
    w_query_fc = tf.get_variable(name +"_w",
                                 trainable=trainable,
                                 initializer=tf.random_uniform([in_size, out_size], -wlimit, wlimit),
                                 regularizer=regularizer)
    b_query_fc = tf.get_variable(name +"_b",
                                 trainable=trainable,
                                 initializer=tf.random_uniform([out_size], -wlimit, wlimit),
                                 regularizer=regularizer)
    if user_bias == False:
      Wx_plus_b = tf.matmul(inputs, w_query_fc)
    else:
      if len(inputs.shape.as_list()) == 3:
        inputs_shape = inputs.shape.as_list()
        temp_input = tf.reshape(inputs, [-1]+inputs_shape[-1:])
        Wx_plus_b = tf.matmul(temp_input, w_query_fc) + b_query_fc
        reshape_tensor = inputs_shape[0:2]+[out_size]
        for i in range(0,len(reshape_tensor)):
          if reshape_tensor[i] == None:
            reshape_tensor[i] = -1
        Wx_plus_b = tf.reshape(Wx_plus_b, reshape_tensor)
      else:
        Wx_plus_b = tf.matmul(inputs, w_query_fc) + b_query_fc

    if activation_function is None:
      outputs = Wx_plus_b
    else:
      if batch_normalize_name is None :
        outputs = activation_function(Wx_plus_b)
      else:
        outputs = activation_function \
          (_batch_norm(Wx_plus_b, out_size, trainable_tensor, batch_normalize_name))
    return outputs


# embedding aggregate
# return [aggreated emb, avg_emb, weight_norm, prob_norm]
def seg_embedding_avg(emb_raw, mask):
  emb_avg = tf.div(tf.reduce_sum(emb_raw, 1),
                     tf.cast(tf.reduce_sum(mask, 1, keep_dims=True), tf.float32))
  emb_aggreage = emb_avg
  return {'emb_aggreate':emb_aggreage, 'emb_avg':emb_avg}

def seg_embedding_avg_selfattention_allsoftmax(emb_raw, mask):
  emb_avg = seg_embedding_avg(emb_raw, mask)['emb_avg']
  attention_weight = \
    tf.reduce_sum(
      tf.multiply(tf.expand_dims(emb_avg, axis=1), emb_raw),
      2,
      keep_dims=True
    )
  attention_weight_norm = tf.norm(tf.squeeze(attention_weight), axis=1)
  attention_prob = \
    tf.nn.softmax(attention_weight, dim=1)
  attention_prob_entropy = tf.reduce_mean(
    tf.reduce_sum(attention_prob*(-tf.log(tf.maximum(attention_prob, 0.00000001))), axis=1)
  )
  emb_aggreate = tf.reduce_sum(emb_raw * attention_prob, axis=1)

  return {'emb_aggreate':emb_aggreate, 'emb_avg':emb_avg, 'attention_weight_norm':attention_weight_norm,
          'attention_prob_entropy':attention_prob_entropy, 'attention_prob':attention_prob}

def seg_embedding_avg_seflattention_maskedsoftmax(emb_raw, mask):
  emb_avg = seg_embedding_avg(emb_raw, mask)['emb_avg']
  # seg_attention_weight [batch, seg, 1]
  seg_attention_weight = \
    tf.reduce_sum(
      tf.multiply(tf.expand_dims(emb_avg, axis=1), emb_raw),
      2,
      keep_dims=True
    )
  attention_mask = tf.expand_dims(mask, axis=2)
  adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
  seg_attention_weight += adder
  attention_prob = tf.nn.softmax(seg_attention_weight, dim=1)
  attention_prob_entropy = tf.reduce_mean(
    tf.reduce_sum(attention_prob*(-tf.log(tf.maximum(attention_prob, 0.00000001))), axis=1)
  )

  emb_aggreate = tf.reduce_sum(emb_raw * attention_prob, axis=1)

  return {'emb_aggreate':emb_aggreate, 'emb_avg':emb_avg, 'attention_prob_entropy':attention_prob_entropy, 'attention_prob':attention_prob}

def seg_embedding_square_selfattention_softmax(emb_raw, mask):
  matmul_mask = tf.matmul(tf.expand_dims(mask, 2), tf.expand_dims(mask, 1))
  # seg_attention_weight [batch, seg, seg]
  seg_attention_weight = \
    tf.reduce_sum(
      tf.matmul(emb_raw, tf.transpose(emb_raw, [0, 2, 1])) * tf.cast(matmul_mask, dtype=tf.float32),
      axis=2,
      keep_dims=True
    )
  attention_mask = tf.expand_dims(mask, axis=2)
  adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
  seg_attention_weight += adder
  seg_attention_prob = tf.nn.softmax(seg_attention_weight, dim=1)
  attention_prob_entropy = tf.reduce_mean(
    tf.reduce_sum(seg_attention_prob*(-tf.log(tf.maximum(seg_attention_prob, 0.00000001))), axis=1)
  )

  emb_aggreate = tf.reduce_sum(emb_raw * seg_attention_prob, axis=1)

  return {'emb_aggreate':emb_aggreate, 'attention_prob_entropy':attention_prob_entropy, 'attention_prob':seg_attention_prob}

def seg_embedding_squre_selfattention_scaledsoftmax(emb_raw, mask):
  dimension =  emb_raw.shape.as_list()[-1]
  matmul_mask = tf.matmul(tf.expand_dims(mask, 2), tf.expand_dims(mask, 1))
  # seg_attention_weight [batch, seg, seg]
  seg_attention_weight = \
    tf.reduce_sum(
      tf.matmul(emb_raw, tf.transpose(emb_raw, [0, 2, 1])) * tf.cast(matmul_mask, dtype=tf.float32),
      axis=2,
      keep_dims=True
    )
  attention_mask = tf.expand_dims(mask, axis=2)
  adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -1000000.0
  seg_attention_weight += adder
  seg_attention_prob = tf.nn.softmax(seg_attention_weight/tf.sqrt(tf.cast(dimension, dtype=tf.float32)), dim=1)
  attention_prob_entropy = tf.reduce_mean(
    tf.reduce_sum(seg_attention_prob*(-tf.log(tf.maximum(seg_attention_prob, 0.00000001))), axis=1)
  )

  emb_aggreate = tf.reduce_sum(emb_raw * seg_attention_prob, axis=1)

  return {'emb_aggreate':emb_aggreate, 'attention_prob_entropy':attention_prob_entropy, 'attention_prob':seg_attention_prob}


def seg_embedding_square_selfattention_shiftsoftmax(emb_raw, mask):
  matmul_mask = tf.matmul(tf.expand_dims(mask, 2), tf.expand_dims(mask, 1))
  # seg_attention_weight [batch, seg, seg]
  seg_attention_weight = \
    tf.reduce_sum(
      tf.matmul(emb_raw, tf.transpose(emb_raw, [0, 2, 1])) * tf.cast(matmul_mask, dtype=tf.float32),
      axis=2,
      keep_dims=True
    )
  attention_mask = tf.expand_dims(mask, axis=2)
  adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
  seg_attention_weight += adder

  seg_attention_prob = tf.nn.softmax(seg_attention_weight - tf.reduce_max(seg_attention_weight, 1, keep_dims=True), dim=1)
  attention_prob_entropy = tf.reduce_mean(
    tf.reduce_sum(seg_attention_prob*(-tf.log(tf.maximum(seg_attention_prob, 0.00000001))), axis=1)
  )
  emb_aggreate = tf.reduce_sum(emb_raw * seg_attention_prob, axis=1)

  return {'emb_aggreate':emb_aggreate, 'attention_prob_entropy':attention_prob_entropy,  'attention_prob':seg_attention_prob}

def seg_embedding_square_selfattention_normsoftmax(emb_raw, mask):
  matmul_mask = tf.matmul(tf.expand_dims(mask, 2), tf.expand_dims(mask, 1))
  emb_norm = emb_raw / tf.norm(emb_raw, axis=2, keep_dims=True)

  # seg_attention_weight [batch, seg, seg]
  seg_attention_weight = \
    tf.reduce_sum(
      tf.matmul(emb_norm, tf.transpose(emb_norm, [0, 2, 1])) * tf.cast(matmul_mask, dtype=tf.float32),
      axis=2,
      keep_dims=True
    )
  attention_mask = tf.expand_dims(mask, axis=2)
  adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
  seg_attention_weight += adder
  seg_attention_prob = tf.nn.softmax(seg_attention_weight, dim=1)
  attention_prob_entropy = tf.reduce_mean(
    tf.reduce_sum(seg_attention_prob*(-tf.log(tf.maximum(seg_attention_prob, 0.00000001))), axis=1)
  )

  emb_aggreate = tf.reduce_sum(emb_raw * seg_attention_prob, axis=1)

  return {'emb_aggreate':emb_aggreate, 'attention_prob_entropy':attention_prob_entropy,  'attention_prob':seg_attention_prob}

def get_emb_aggregate_func(num):
  fs = [seg_embedding_avg, seg_embedding_avg_selfattention_allsoftmax, seg_embedding_avg_seflattention_maskedsoftmax,
        seg_embedding_square_selfattention_softmax, seg_embedding_squre_selfattention_scaledsoftmax, seg_embedding_square_selfattention_shiftsoftmax,
        seg_embedding_square_selfattention_normsoftmax]
  return fs[num]

def get_bert_pooled_output(bert, mask):
  first_token_emb = bert.get_first_token_output()

  bert_last_layer_masked_avg = bert.get_sequence_output() * tf.cast(tf.expand_dims(mask, dim=2), dtype=tf.float32)
  bert_last_layer_masked_max = bert.get_sequence_output() * tf.cast(tf.expand_dims(mask, dim=2), dtype=tf.float32) - tf.cast((1 - tf.expand_dims(mask, dim=2))*999999999, dtype=tf.float32)
  bert_last_layer_masked_min = bert.get_sequence_output() * tf.cast(tf.expand_dims(mask, dim=2), dtype=tf.float32) + tf.cast((1 - tf.expand_dims(mask, dim=2))*999999999, dtype=tf.float32)

  bert_avg_pooling = tf.div(tf.reduce_sum(bert_last_layer_masked_avg, 1),
                                  tf.cast(tf.reduce_sum(mask, 1, keep_dims=True), tf.float32))
  bert_max_pooling = tf.reduce_max(bert_last_layer_masked_max, axis=1)
  bert_min_pooling = tf.reduce_min(bert_last_layer_masked_min, axis=1)

  return tf.concat([first_token_emb, bert_avg_pooling, bert_max_pooling, bert_min_pooling], axis=1)

# def _get_pooled_output(self, bert, mask):
#   first_token_emb = bert.get_first_token_output()
#
#   bert_last_layer_masked_avg = bert.get_sequence_output() * tf.cast(tf.expand_dims(mask, dim=2), dtype=tf.float32)
#   bert_last_layer_masked_max = bert.get_sequence_output() * tf.cast(tf.expand_dims(mask, dim=2), dtype=tf.float32) - tf.cast((1 - tf.expand_dims(mask, dim=2))*999999999, dtype=tf.float32)
#   bert_last_layer_masked_min = bert.get_sequence_output() * tf.cast(tf.expand_dims(mask, dim=2), dtype=tf.float32) + tf.cast((1 - tf.expand_dims(mask, dim=2))*999999999, dtype=tf.float32)
#
#   bert_avg_pooling = tf.div(tf.reduce_sum(bert_last_layer_masked_avg, 1),
#                                   tf.cast(tf.reduce_sum(mask, 1, keep_dims=True), tf.float32))
#   bert_max_pooling = tf.reduce_max(bert_last_layer_masked_max, axis=1)
#   bert_min_pooling = tf.reduce_min(bert_last_layer_masked_min, axis=1)
#
#   return tf.concat([first_token_emb, bert_avg_pooling, bert_max_pooling, bert_min_pooling], axis=1)
