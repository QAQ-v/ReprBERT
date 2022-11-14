#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import logging
import os.path
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

"""
auc 目的是求ROC曲线的面积
"""
class auc():
  def __init__(self,scores,labels):
    self.scores = scores
    self.labels = labels
  def getArea(self,x1,x2,y1,y2): # 求矩阵\梯形面积
    return (x1-x2)*(y1+y2)/2.0 # 上底加下底乘高除以二
  def isEqual(self,x,y):
    if(np.abs(x-y)<np.exp(-8)):
      return 1
    else:
      return 0
  def getAuc(self):

    score = roc_auc_score(self.labels, self.scores)
    return score.item()

    # area=tp=fp=tppv=fppv=0
    # prob=-65535
    # sortedScores=np.sort(self.scores)[::-1]#对scores进行降序排列
    # sortedIndex=np.argsort(self.scores)[::-1]#对scores降序排列对应的原始数据位置序列
    # P=np.count_nonzero(self.labels)#正类个数
    # N=len(self.scores)-P#负类个数
    # for i in np.arange(len(self.labels)):
    #   if(self.isEqual(sortedScores[i],prob)==0):
    #     area+=self.getArea(fp,fppv,tp,tppv)
    #     prob=sortedScores[i]
    #     tppv=tp
    #     fppv=fp
    #   if (self.labels[sortedIndex[i]]==1):
    #     tp+=1
    #   else:
    #     fp+=1
    # area += self.getArea(N,fppv,P,tppv)
    # #rint N
    # #print P
    # if N*P == 0:
    #   return -1.0
    # else:
    #   return area/float(N*P)


# label = []
# score = []
# for line in sys.stdin:
#   sp = line.split('\t')
#   label.append(int(sp[2]))
#   score.append(float(sp[3]))
#
# a=auc(score, label)
# print a.getAuc()
# get the auc 0.748 for savedmodel