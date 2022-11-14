#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import sys
import tensorflow as tf
import numpy as np
sys.path.append('../')
from common.input.tf_input import *
from model.factory import *
from common.util.util import *

class Evaluation(object):

  def __init__(self, model_name, input_config):
    self.model_name = model_name
    self.input_config = input_config
    self.input_tag = input_config['table_name'].split('/')[-1]


  def _build_input(self):
    self.input = input_table_batch_fn(self.input_config['slice_count'],
                                      self.input_config['slice_id'],
                                      self.input_config['table_name'],
                                      self.input_config['batch_size'],
                                      self.input_config['epoch'],
                                      self.input_config['columns'],
                                      'input_table',
                                      False)

  def _build_summary(self):
    self.metric_summary_ph_dict = {}
    self.metric_summary_ops = []
    with tf.name_scope(name=self.input_tag):
      for k, v in self.metric_summary_names.items():
        self.metric_summary_ph_dict[k] = tf.placeholder(tf.float32)
        if v == 'scalar':
          self.metric_summary_ops.append(tf.summary.scalar(k, self.metric_summary_ph_dict[k]))
        elif v == 'histogram':
          self.metric_summary_ops.append(tf.summary.histogram(k, self.metric_summary_ph_dict[k]))

  def _build_eval_op(self, reuse, model_config):
    model_config.batch_size = self.input_config['batch_size']

  def _run_session(self, session, step, global_step):
    pass

  def _feed_summary(self, sess, summary_writer_monitoredtrainingsession, global_step):
    feed_dict = {}
    for k in self.metric_summary_names.keys():
      feed_dict[self.metric_summary_ph_dict[k]] = self.metric_summary_res[k]
    summary_metric_list = sess.run(self.metric_summary_ops,
                                   feed_dict=feed_dict)
    for e in summary_metric_list:
      summary_writer_monitoredtrainingsession.add_summary(e, global_step)



class PointwiseRichEvaluation(Evaluation):

  def __init__(self, model_name, input_config):
    super(PointwiseRichEvaluation, self).__init__(model_name, input_config)



  def _build_summary(self):
    # self.metric_summary_names = {'precision_@20': 'scalar', 'precision_@50': 'scalar',
    #                         'precision_topk_precision': 'histogram',
    #                         'auc_avg': 'scalar', 'auc_all': 'scalar', 'streaming_auc': 'scalar'}
    self.metric_summary_names = {'auc_all': 'scalar', 'streaming_auc': 'scalar'}
    super(PointwiseRichEvaluation, self)._build_summary()


  def _build_eval_op(self, reuse, model_config):
    model_config.batch_size = self.input_config['batch_size']
    with tf.variable_scope("network", reuse=reuse):
      self.model = model = ModelFactory(self.model_name, model_config, self.input)
    with tf.variable_scope("my_streaming"):
      self.stream_auc = tf.contrib.metrics.streaming_auc(tf.sigmoid(model.pos_distance), # for cosine scalar
      # self.stream_auc = tf.contrib.metrics.streaming_auc(model.pos_distance, # for entropy
                                                    tf.reshape(self.input['label'],
                                                               [self.input_config['batch_size']]),
                                                    name='stream_auc')
      self.test_relevence_input_query = tf.reshape(self.input['query'], [-1])
      self.test_relevence_input_nid = tf.reshape(self.input['nid'], [-1])
      self.test_relevence_input_label = tf.reshape(self.input['label'], [-1])
      self.pos_distance = model.pos_distance


  def _run_session(self, session, step, global_step):
    total_count = self.input_config['total_count']
    if total_count == -1:
      iters = 1
    else:
      iters = total_count/self.input_config['batch_size']

    eval_index = None
    eval_label = None
    eval_dis = None
    eval_nid = None
    valid_init_flag = False
    total_run = 0
    idx = 0
    while (idx < iters): # 每次run一个batch, 然后把这些batch累加起来, 就实现了对整个数据集的预测
      r1, r4, r2, r3, s_auc = session.run(
                    [self.test_relevence_input_query,
                     self.test_relevence_input_nid,
                     self.test_relevence_input_label,
                     self.pos_distance,
                     self.stream_auc])
      # if idx == 0:
      #   print "sample predict data"
      #   print r2
      #   print r3

      if not valid_init_flag:
        eval_index = np.array(r1)
        eval_label = np.array(r2)
        eval_dis = np.array(r3)
        eval_nid = np.array(r4)
        valid_init_flag = True
      else:
        eval_index = np.append(eval_index, r1)
        eval_label = np.append(eval_label, r2)
        eval_dis = np.append(eval_dis, r3)
        eval_nid = np.append(eval_nid, r4)
      idx = idx + 1
      total_run = total_run + len(r1)
    print np.mean(eval_label), np.max(eval_label), np.min(eval_label)
    print np.mean(eval_dis), np.max(eval_dis), np.min(eval_dis)
    # if step % 20 == 3:
    #   print eval_label
    #   print eval_dis
    # top_k1_precision, top_k2_precision, topk1_precision, auc_avg, all_auc = \
    #   evaluate(step, global_step, eval_index, eval_nid, eval_label, eval_dis, eval_index.size, False)
    all_auc = \
      evaluate(step, global_step, eval_index, eval_nid, eval_label, eval_dis, eval_index.size, False)
    print(time_stamp(), "streaming auc for rel test data: %s" % (str(s_auc)))
    print(time_stamp(), "Finish to predict the evaluate relevence data, iter: %d, total: %d" % (idx, total_run))

    self.metric_summary_res = {}
    # self.metric_summary_res['precision_@20'] = top_k1_precision
    # self.metric_summary_res['precision_@50'] = top_k2_precision
    # self.metric_summary_res['precision_topk_precision'] = topk1_precision
    # self.metric_summary_res['auc_avg'] = auc_avg
    self.metric_summary_res['auc_all'] = all_auc
    self.metric_summary_res['streaming_auc'] = s_auc[0]

class PointwiseAUCEvaluation(Evaluation):
  def __init__(self, model_name, input_config):
    super(PointwiseAUCEvaluation, self).__init__(model_name, input_config)

  def _build_summary(self):
    self.metric_summary_names = {'streaming_auc': 'scalar'}
    super(PointwiseAUCEvaluation, self)._build_summary()


  def _build_eval_op(self, reuse, model_config):
    model_config.batch_size = self.input_config['batch_size']
    with tf.variable_scope("network", reuse=reuse):
      self.model = model = ModelFactory(self.model_name, model_config, self.input)
    with tf.variable_scope("my_streaming"):
      self.stream_auc = tf.contrib.metrics.streaming_auc(tf.sigmoid(model.pos_distance),
                                                    tf.reshape(self.input['label'],
                                                               [self.input_config['batch_size']]),
                                                    name='stream_auc')



  def _run_session(self, session, step, global_step):
    total_count = self.input_config['total_count']
    if total_count == -1:
      iters = 1
    else:
      iters = total_count/self.input_config['batch_size']

    while iters > 0:
      stream_auc = session.run([self.stream_auc])
      iters -= 1

    print(time_stamp(), "streaming auc for %s data: %s" % (self.input_tag, str(stream_auc)))

    self.metric_summary_res = {}
    self.metric_summary_res['streaming_auc'] = stream_auc[0][0]

class PvwiseMrrEvaluation(Evaluation):
  def __init__(self, model_name, input_config):
    super(PvwiseMrrEvaluation, self).__init__(model_name, input_config)

  def _build_summary(self):
    self.metric_summary_names = {'zuoshumrr':'scalar','pvmrr':'scalar'}
    super(PvwiseMrrEvaluation, self)._build_summary()

  def _build_eval_op(self, reuse, model_config):
    model_config.batch_size = self.input_config['batch_size']
    with tf.variable_scope("network", reuse=reuse):
      self.input['pos_t_word_char_id'] = tf.reshape(self.input['pos_t_word_char_id'], [-1, 30])
      model_mrr = ModelFactory(self.model_name, model_config, self.input)
      query_rep = model_mrr.query_rep
      docs_rep = model_mrr.doc_rep

      self.eval_rank = \
        tf.reduce_sum(tf.multiply(tf.expand_dims(tf.transpose(query_rep), 1), tf.reshape(tf.transpose(docs_rep), [-1, 11, 128])), 2,
                      keep_dims=False)
      self.click_pos = self.input['click_pos']
      self.model = model_mrr

  def _run_session(self, session, step, global_step):
    total_count = self.input_config['total_count']
    if total_count == -1:
      iters = 1
    else:
      iters = total_count/self.input_config['batch_size']
    idx = 0
    mrr_array = []
    mrr_pv = []
    while idx < iters:
      r, c = session.run([self.eval_rank, self.click_pos])
      l = len(c)
      j = 0
      while j < l:
        score = 1.0
        for index, v in enumerate(r[j]):
          if index != c[j] and v > r[j][c[j]]:
            score = score + 1.0
        mrr_array.append(1.0 / score)
        mrr_pv.append(1.0 / (c[j] + 1))
        j = j + 1
      idx = idx + 1
    mrr_avg = np.mean(np.array(mrr_array))
    mrr_pv_value = np.mean(np.array(mrr_pv))
    print time_stamp(), "%s mrr value: %lf, %lf" % (self.input_tag, mrr_avg, mrr_pv_value)

    self.metric_summary_res = {}
    self.metric_summary_res['zuoshumrr'] = mrr_avg
    self.metric_summary_res['pvmrr'] = mrr_pv_value
    return mrr_avg, mrr_pv_value

def CreateEvaluation(name, model_name, input_config):
  if name == 'PointwiseAUCEvaluation':
    return PointwiseAUCEvaluation(model_name, input_config)
  elif name == 'PointwiseRichEvaluation':
    return PointwiseRichEvaluation(model_name, input_config)
  elif name == 'PvwiseMrrEvaluation':
    return PvwiseMrrEvaluation(model_name, input_config)
  else:
    raise ValueError("Not supportted evaluation: %s" % name)