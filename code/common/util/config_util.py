#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import tensorflow as tf
import json
import six
import copy
import re


class ModelConfig(object):
  def __init__(self,
              FLAGS,
              trainable=True):
    self.FLAGS=FLAGS
    self.trainable=trainable


  @classmethod
  def from_dict(cls, F, json_object):
    """Constructs a `ModelConfig` from a Python dictionary of parameters."""
    config = ModelConfig(FLAGS=F)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, FLAGS, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(FLAGS, json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



instance_config = \
  {
    "relevence_train": \
      {
        'trainable': True,
        #'batch_size': 128,
      },
    "relevence_test": \
      {
        'trainable': False,
        #'batch_size': 128,
      },
    "rankservice_dump_label_test": \
      {
        'trainable': False,
        #'batch_size': 128,
      },
  }



class customize_config(object):

  def __init__(self):
    flags = tf.app.flags

    flags.DEFINE_string('input_column_conf', 'conf/input_conf/input_feature_column.json', '')
    flags.DEFINE_string('input_table_conf', 'conf/input_conf/input_table.json', '')
    flags.DEFINE_string('input_model_block_conf', 'conf/input_conf/test_graph_1.json', '')
    flags.DEFINE_string('table_columns', '', '')

    flags.DEFINE_string("tables", "odps://search_offline_dev/tables/recall_relevence_data_positive_feature_v4_data/type=leafcate",  "tables info")
    flags.DEFINE_string('outputs', '', '')
    flags.DEFINE_integer("task_index", None, "Worker task index")
    flags.DEFINE_string("ps_hosts", "", "ps hosts")
    flags.DEFINE_string("worker_hosts", "", "worker hosts")
    flags.DEFINE_string("job_name", None, "job name: worker or ps")

    flags.DEFINE_integer("batch_size", 512, "")
    flags.DEFINE_integer("batch_size_test_relevence", 19093, "")
    flags.DEFINE_integer("batch_size_test_userlog", 100000, "")
    flags.DEFINE_integer("batch_size_test", 512, "")
    flags.DEFINE_integer("query_length", 13, "")
    flags.DEFINE_integer("doc_length", 30, "")
    flags.DEFINE_integer("img_size", 2048, "")
    flags.DEFINE_integer("vocab_size", 1105039, "")
    flags.DEFINE_integer("embedding_size", 100, "")
    flags.DEFINE_integer("num_epoches", 10000000000000, "")
    flags.DEFINE_integer("nid_code_size", 2851797, "")
    flags.DEFINE_integer("seller_code_size", 3000001, "")
    flags.DEFINE_integer("cate_code_size", 14140, "")
    flags.DEFINE_integer("brand_code_size", 1000001, "")
    flags.DEFINE_integer("data_size", 2000000, "")

    flags.DEFINE_float("weight_decay", 0.000000001, "")
    flags.DEFINE_float("margin", 0.2, "")
    flags.DEFINE_float("learning_base", 0.001, "")
    flags.DEFINE_float("max_grad_norm", 2, "")
    flags.DEFINE_float("learning_rate_decay", 0.1, "")
    flags.DEFINE_float("decay_step", 5000000, "")
    flags.DEFINE_float("moving_average_decay", 0.9, "")

    flags.DEFINE_string("checkpointDir", "", "")
    flags.DEFINE_string("pb_file_path", "oss://zuoshu/shaowei-ysw/", "")
    flags.DEFINE_string("checkpointBaseDir", "oss://haochuantf/deep_match/checkpoint1/", "")
    flags.DEFINE_string("checkpointMetricDir", "oss://haochuantf/deep_match/metric/", "")
    flags.DEFINE_string("ModelTag", "", "")
    flags.DEFINE_bool("initw2v", True, "")
    flags.DEFINE_string("model_name", "model_fullcate_CNN_words", "")
    flags.DEFINE_string("input_name", "query_title_ids_word", "input function name")
    flags.DEFINE_integer("rel_query_index", 5, "")
    flags.DEFINE_integer("rel_label_index", 7, "")
    flags.DEFINE_bool("sync", False, "")
    flags.DEFINE_integer("evaluation_worker_index", 1, "")
    flags.DEFINE_integer("initw2v_worker_index", 0, "")
    flags.DEFINE_integer("save_checkpoint_secs", 3600, "")

    flags.DEFINE_bool("timeline", False, "")
    flags.DEFINE_string("saver_list", "", "")

    flags.DEFINE_integer("output_dim", 128, "output_dim")
    flags.DEFINE_float('grl_total_step', 1.0, '')

    flags.DEFINE_bool("resnet_infer_trainable", True, "")
    flags.DEFINE_bool("dssm_trainable", True, "")
    flags.DEFINE_bool("resnet_project_trainable", True, "")

    flags.DEFINE_string("instance_list", "relevence_train", "")

    flags.DEFINE_string("image_dataformat", None, "channels_last for cpu")
    flags.DEFINE_integer("partition_num", 10, '')

    flags.DEFINE_integer('window_move_step', 200000,'')
    flags.DEFINE_float('window_nparts', 5.0, '')
    flags.DEFINE_float('window_size', 0.2, '')
    flags.DEFINE_integer('periodic_step', 10, '')
    flags.DEFINE_integer('peroid_limit', 3, '')
    flags.DEFINE_float('stable_margin', 0.18, '')

    flags.DEFINE_integer('emb_aggregate_func_num', 0, '')

    flags.DEFINE_integer('bert_vocab_size', 21128, '')
    flags.DEFINE_integer('bert_hidden_size',  768, '')
    flags.DEFINE_integer('bert_num_hidden_layers', 4, '')
    flags.DEFINE_integer('bert_num_attention_heads', 12, '')
    flags.DEFINE_integer('bert_intermediate_size', 3072, '')
    flags.DEFINE_string('bert_hidden_act', 'gelu', '')
    flags.DEFINE_float('bert_hidden_dropout_prob', 0.1, '')
    flags.DEFINE_float('bert_attention_probs_dropout_prob', 0.1, '')
    flags.DEFINE_integer('bert_max_position_embeddings', 512, '')
    flags.DEFINE_integer('bert_type_vocab_size', 2, '')
    flags.DEFINE_float('bert_initializer_range', 0.02, '')
    flags.DEFINE_bool('bert_if_use_first_token', True, '')

    flags.DEFINE_integer('loss_type', 0, '')
    flags.DEFINE_bool('embedding_trainable', True, '')

    flags.DEFINE_bool('pai_soar', False, '')


    #flags.DEFINE_string('infer_output_table', '', '')
    flags.DEFINE_string('infer_switch', "", '')


    self.FLAGS = tf.app.flags.FLAGS

    with open(self.FLAGS.input_column_conf, 'r') as f:
      self.input_column_config = json.load(f)
      for k,v in self.input_column_config.items():
        v['dtype']=eval(v['dtype'])
    with open(self.FLAGS.input_table_conf, 'r') as f:
      self.input_config = json.load(f)


    self.model_instance = []
    for e in self.FLAGS.instance_list.split(','):
      if len(e) > 0:
        mc = ModelConfig.from_dict(self.FLAGS, instance_config[e])
        mc.batch_size = self.FLAGS.batch_size
        self.model_instance.append(mc)
    print "GET model instance %d" % (len(self.model_instance))

    self.input_instance = []
    match_pattern = re.compile(r"(=\d{8})")
    replace_pattern = re.compile(r"(\d{8})")
    table_columns=None
    if len(self.FLAGS.table_columns) > 0:
      assert (len(self.FLAGS.table_columns.split(';')) == len(self.FLAGS.tables.split(',')))
      table_columns=[e.split(',') for e in self.FLAGS.table_columns.split(';')]
    # 每个表就是一个instances
    for idx, e in enumerate(self.FLAGS.tables.split(',')): 
      if len(e) > 0:
        input_table = e
        date_match = re.search(match_pattern, e)
        if date_match and  len(date_match.groups()) > 0:
          input_table = re.sub(replace_pattern, "*", e)
        input_config_object = self.input_config[input_table]
        input_config_object['table_name'] = e
        if table_columns:
          input_config_object['columns']=table_columns[idx]
        temp = []
        for c in input_config_object['columns']:
          if len(c) > 0:
            temp.append(self.input_column_config[c])
        input_config_object['columns'] = temp
        if input_config_object['train'] == True:
          input_config_object['batch_size'] = self.FLAGS.batch_size
        if self.FLAGS.pai_soar == False and input_config_object['train'] == True and 'slice_count' not in input_config_object and len(self.FLAGS.tables.split(',')) > 1:
          input_config_object['slice_count'] = len(self.FLAGS.worker_hosts.split(',')) - 1
          input_config_object['slice_id'] = self.FLAGS.task_index
        elif 'slice_count' not in input_config_object:
          input_config_object['slice_count'] = len(self.FLAGS.worker_hosts.split(','))
          input_config_object['slice_id'] = self.FLAGS.task_index

        self.input_instance.append(input_config_object)
    print "GET input instance %d" % (len(self.input_instance))


# a=customize_config()
# print a.model_instance
# print a.input_instance


