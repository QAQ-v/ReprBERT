#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import numpy as np
import time
from auc_calculate import auc

def time_stamp():
  return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

def evaluate(step, a, eval_index, eval_nid, eval_label, eval_dis, batch_size_test, iflog):
  score_array = np.array(eval_dis)
  score_rank_idx = np.argsort(score_array)
  if iflog == True:
    false_negtive_cnt = 0
    for idx in score_rank_idx:
      if eval_label[idx] == 1:
        print "false negtive: %f %s %Ld %d" % (eval_dis[idx], eval_index[idx], eval_nid[idx], eval_label[idx])
        false_negtive_cnt = false_negtive_cnt + 1
        if (false_negtive_cnt > 10):
          break

    false_positive  = 0
    for i in range(0, score_rank_idx.__len__())[::-1]:
      idx = score_rank_idx[i]
      if eval_label[idx] == 0:
        print "false positive: %f %s %Ld %d" % (eval_dis[idx], eval_index[idx], eval_nid[idx], eval_label[idx])
        false_positive = false_positive + 1
        if false_positive > 10:
          break



  dict_dist = {}
  topk1_precision = []
  topk2_precision = []
  auc_list = []
  topk1 = 10
  topk2 = 20
  all_auc_measure = auc(np.array(eval_dis), np.array(eval_label)) # all auc就是这个batch下所有query-item pair的auc
  all_auc = all_auc_measure.getAuc()
  # for i in range(batch_size_test): # 一个test 的 batch size !! 其实是整个testset的大小!
  #   if eval_index[i] not in dict_dist.keys(): # eval index 的key是每个query
  #     dict_dist.setdefault(eval_index[i], [])
  #   dict_dist[eval_index[i]].append((eval_dis[i], eval_label[i]))
  # for key in dict_dist: 
  #   values = dict_dist[key]
  #   sorted_value = sorted(values, key=lambda x: x[0], reverse=True)
  #   scores = [x[0] for x in sorted_value]
  #   labels = [x[1] for x in sorted_value]
  #   meauser_auc = auc(np.array(scores), np.array(labels))
  #   auc_score = meauser_auc.getAuc()
  #   if auc_score > 0:
  #     auc_list.append(auc_score) # auc_list存的是每个query下的所有商品的auc
  #   sum = 0
  #   min_topk2 = min(topk2, len(values))
  #   min_topk1 = min(topk1, len(values))
  #   for j in range(min_topk2):
  #     sum = sum + sorted_value[j][1]
  #     if j == min_topk1 - 1:
  #       topk1_precision.append(float(sum) / min_topk1)
  #   topk2_precision.append(float(sum) / min_topk2)
  # top_k1_precision = np.mean(np.array(topk1_precision))
  # top_k2_precision = np.mean(np.array(topk2_precision))
  # auc_avg = np.mean(np.array(auc_list))
  print (time_stamp(), "Finished calculate the precision, length: %d !" % (batch_size_test))
  # print ("%s Evaluation Step %d / %d, p10 : %f, p20 : %f, group AUC : %f, all AUC : %f" % (
  # time_stamp(), step, a, top_k1_precision, top_k2_precision, auc_avg, all_auc))
  print ("%s Evaluation Step %d / %d, all AUC : %f" % (time_stamp(), step, a, all_auc))

  # return top_k1_precision, top_k2_precision, topk1_precision, auc_avg, all_auc
  return all_auc

def mrr(distance_matrix, userlog_input_query, userlog_input_nid, iflog):
  dim1,dim2 = distance_matrix.shape

  i = 0
  scores = []
  query_badcase=[]
  while i < dim1:
    badcase=[]
    j = 0
    target = distance_matrix[i, i]
    rank = 0
    while j < dim2:
      if i != j and distance_matrix[i, j] >= target:
        rank = rank + 1
        if iflog == True:
          badcase.append(userlog_input_nid[j])
      j = j + 1
    scores.append(1/float(rank+1))
    query_badcase.append(badcase)
    i = i + 1

  score_array = np.array(scores)
  # if iflog == True:
  #   query_score_sort_idx = np.argsort(-score_array)
  #   i = 0
  #   while i < 20:
  #     print scores[query_score_sort_idx[i]], score_array[query_score_sort_idx[i]], str(query_badcase[query_score_sort_idx[i]])
  #     i = i + 1

  return np.mean(score_array)



