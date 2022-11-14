#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import tensorflow as tf


def ModelFactory(name, config, input):
  mo = None

  if name == 'model_query_title_cnn':
    from model_query_title_cnn import *
  elif name == 'model_query_title_avg_fc':
    from model_query_title_avg_fc import *
  elif name == 'model_query_title_avg_fc_bn_dropout':
    from model_query_title_avg_fc_bn_dropout import *
  elif name == 'model_query_title_fbcnn':
    from model_query_title_fbcnn import *
  elif name == 'model_query_title_bert':
    from model_query_title_bert import *
  elif name == 'model_query_title_avg_fc_dynamic_margin':
    from model_query_title_avg_fc_dynamic_margin import *
  elif name == 'model_query_title_selfattention_maskedsoftmax_fc':
    from model_query_title_selfattention_maskedsoftmax_fc import *
  elif name == 'model_query_title_selfattention_allsoftmax_fc':
    from model_query_title_selfattention_allsoftmax_fc import *
  elif name == 'model_query_title_selfattention_squaresoftmax_fc':
    from model_query_title_selfattention_squaresoftmax_fc import *
  elif name == 'model_query_title_selfattention_adapter_fc':
    from model_query_title_selfattention_adapter_fc import *
  elif name == 'model_query_title_avg_fc_term_rank':
    from model_query_title_avg_fc_term_rank import *
  elif name == 'model_query_title_bert_with_emb':
    from model_query_title_bert_with_emb import *
  elif name == 'model_query_title_avg_fc_single':
    from model_query_title_avg_fc_single import *
  elif name == 'model_query_title_avg_fc_pair':
    from model_query_title_avg_fc_pair import *
  elif name == 'model_query_title_pooling_fc':
    from model_query_title_pooling_fc import *
  elif name == 'model_query_title_bert_pooling':
    from model_query_title_bert_pooling import *
  elif name == 'model_query_title_bert_pooling_pair':
    from model_query_title_bert_pooling_pair import *
  elif name == 'model_block_builder':
    from model_block_builder import *
  elif name == 'ModelQueryTitleBertMultiTaskNQPPretrained_SpecificFcAgreementWeight511_Pairwise':
    from ModelQueryTitleBertMultiTaskNQPPretrained_SpecificFcAgreementWeight511_Pairwise import *
  elif name == 'ModelQueryTitleBertMultiTaskNQPPretrained_SpecificFc_Pairwise':
    from ModelQueryTitleBertMultiTaskNQPPretrained_SpecificFc_Pairwise import *
  elif name == 'ModelQueryTitleBertMultiTaskNQPPretrained_SpecificFcAgreementWeight511_Pairwise_Bugfix':
    from ModelQueryTitleBertMultiTaskNQPPretrained_SpecificFcAgreementWeight511_Pairwise_Bugfix import *
  elif name == 'ModelQueryTitleBertMultiTaskNQPPretrained_SpecificFcAgreementWeight511_late':
    from ModelQueryTitleBertMultiTaskNQPPretrained_SpecificFcAgreementWeight511_late import *
  elif name == "ModelQueryTitleBertMultiTaskNQPPretrained_SpecificFcAgreementWeight511_selfattn":
    from ModelQueryTitleBertMultiTaskNQPPretrained_SpecificFcAgreementWeight511_selfattn import *
  elif name == "TwoTowerMyBert":
    from TwoTowerMyBert import *
  elif name == "TwoTowerMyBert_mid":
    from TwoTowerMyBert_mid import *
  elif name == "TwoTowerMyBert_mid_pool":
    from TwoTowerMyBert_mid_pool import *
  elif name == "TwoTowerMyBert_mid_pool_online":
    from TwoTowerMyBert_mid_pool_online import *
  elif name == "TwoTowerMyBert_mid_pool_online_serving":
    from TwoTowerMyBert_mid_pool_online_serving import *
  elif name == "TwoTowerMyBert_mid_pool_online_query_exporter":
    from TwoTowerMyBert_mid_pool_online_query_exporter import *
  elif name == "TwoTowerMyBert_mid_pool_online_title_exporter":
    from TwoTowerMyBert_mid_pool_online_title_exporter import *
  elif name == "TwoTowerMyBert_poly":
    from TwoTowerMyBert_poly import *
  elif name == "TwoTowerMyBert_mid_pool_online_cosine":
    from TwoTowerMyBert_mid_pool_online_cosine import *
  elif name == "TwoTowerMyBert_mid_pool_online_cosine_crossE":
    from TwoTowerMyBert_mid_pool_online_cosine_crossE import *
  elif name == "TwoTowerMyBert_mid_pool_online_newInteraction":
    from TwoTowerMyBert_mid_pool_online_newInteraction import *
  else:
    raise ValueError("Unsupported model: %s" % name)

  mo = Model(config)
  mo.model_fn_train(input)
  return mo


def CreateModel(name, input_list, model_config_list):
  with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
    m_train = ModelFactory(name, model_config_list[0], input_list[0])
  with tf.variable_scope("network", reuse=True):
    m_test_relevence = ModelFactory(name, model_config_list[1], input_list[1])
  with tf.variable_scope("my_streaming"):
    stream_auc = tf.contrib.metrics.streaming_auc(tf.maximum(tf.minimum(m_test_relevence.pos_distance, 1.0), 0.0),
                                                     tf.reshape(input_list[1]['label'], [model_config_list[1].batch_size]))

  return [m_train, m_test_relevence, stream_auc]

