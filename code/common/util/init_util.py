#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import sys
import tensorflow as tf


def CreatePartitionedVariableSaver(prefix):
  res = {}
  for v in tf.global_variables(scope=prefix): 
    name_arr = v.op.name.split("/")
    main_name = "/".join(_ for _ in name_arr if not _.startswith("part_"))
    print "name_arr =", name_arr
    print "main_name =", main_name
    if main_name not in res:
      res[main_name] = []
    res[main_name].append(v)

  _res = {}
  print "%s variable start!!!!!!!!!" % (prefix)
  for k in res: 
    print k
    if len(res[k]) == 1:
      _res[k] = res[k][0]
    else:
      _res[k] = res[k]
  return tf.train.Saver(_res, max_to_keep=1000)

def CreatePartitionedVariableSaverForDifferentPrefix(prefix, target_prefix):  # network/bert_layer/bert/, bert/

  res = {}
  for v in tf.global_variables(scope=prefix):
    name_arr = v.op.name.split("/")
    main_name = "/".join(_ for _ in name_arr if not _.startswith("part_"))
    print "name_arr =", name_arr
    print "main_name =", main_name
    ''' 比如:
    name_arr = [u'network', u'bert_layer', u'bert', u'encoder', u'layer_8', u'output', u'dense', u'bias', u'part_0']
    main_name = network/bert_layer/bert/encoder/layer_8/output/dense/bias  ## 把part_xx都合并到一起'''

    target_name = target_prefix.strip("/") +  main_name[len(prefix.strip("/")):]
    target_name = target_name.strip("/")
    print "target_name =", target_name

    # # rename checkpoint 跨层读取
    # begin = target_name.find('layer')
    # if begin != -1:
    #   target_name = target_name[:begin] + 'layer_' + str(int(target_name[begin+6])*3) + target_name[begin+7:]
    #   print "new target_name =", target_name

    if target_name not in res:
      res[target_name] = []
    res[target_name].append(v)

  _res = {}
  print "%s variable start!!!!!!!!!" % (prefix)
  for k in res:
    print k 
    if len(res[k]) == 1:
      _res[k] = res[k][0]
    else:
      _res[k] = res[k]
  return tf.train.Saver(_res, max_to_keep=1000)

def CreateSaverListFromConfig(config):
  saver_list = []
  for e in config.split(';'):
    sp = e.split(',')
    ckpt = sp[0]
    print "ckpt_path_1", ckpt
    if ckpt.endswith("/*"):
      directory_list = tf.gfile.ListDirectory(ckpt[0:-1])
      if len(directory_list) == 0:
        raise ValueError("Empty directory: %s" % ckpt[0:-1])
      print "directory_list", directory_list
      directory_list.sort()
      last_ckpt = ckpt[0:-1] + directory_list[-1]
      print "init from path: %s" % (last_ckpt)
      if not last_ckpt.endswith('/'):
        last_ckpt = last_ckpt[0:-1] + last_ckpt + '/'
      ckpt = last_ckpt

    # if ckpt.split("/")[-1].find("model.ckpt") == -1:
    if ckpt.endswith("/"):
      ckpt = tf.train.latest_checkpoint(ckpt)
      print "ckpt_path_2", ckpt

    print "ckpt: %s" % (ckpt)
    for p in sp[1:]:
      source_prefix, target_prefix = p.split(':')

      if not source_prefix or len(source_prefix) == 0:
        print "ERROR for CreatePartitionedVariableSaverForDifferentPrefix"
        sys.exit(0)

      if source_prefix == target_prefix:
        saver_list.append((CreatePartitionedVariableSaver(source_prefix), ckpt)) # network/bert_layer/bert/, bert/
      else:
        saver_list.append((CreatePartitionedVariableSaverForDifferentPrefix(source_prefix, target_prefix), ckpt))
  return saver_list




