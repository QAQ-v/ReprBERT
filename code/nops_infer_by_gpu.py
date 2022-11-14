# coding=UTF-8
import os
import tensorflow as tf
import time
from common.input.tf_input import *
from common.util.config_util import *
from model.factory import *
from common.util.init_util import *

global_config=customize_config()
FLAGS = global_config.FLAGS
def print_global_vairable():
  variables = tf.global_variables()
  print "total global variable: %d" %(len(variables))
  for v in variables:
    print "name:%s, shape:%s, op.name:%s"%(v.name, v.shape, v.op.name)

def print_local_vairable():
  variables = tf.local_variables()
  print "total local variable: %d" % (len(variables))
  for v in variables:
    print "name:%s, shape:%s, op.name:%s" % (v.name, v.shape, v.op.name)

def time_stamp():
  return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

def input_padding():
  fake_data={}
  fake_data['ids']=tf.placeholder(tf.int32, shape=[None, 13], name='ids')
  fake_data['mask']=tf.placeholder(tf.int32, shape=[None, 13], name='mask')
  fake_data['seg_ids']=tf.placeholder(tf.int32, shape=[None, 13], name='seg_ids')
  fake_data['ids1']=tf.placeholder(tf.int32, shape=[None, 30], name='ids1')
  fake_data['mask1']=tf.placeholder(tf.int32, shape=[None, 30], name='mask1')
  fake_data['seg_ids1']=tf.placeholder(tf.int32, shape=[None, 30], name='seg_ids1')
  fake_data['nid']=tf.placeholder(tf.int64, shape=[None, 1], name='nid')
  fake_data['query'] = tf.placeholder(tf.string, shape=[None, 1])
  return fake_data


def train(worker_count, ps_count, task_index, cluster, is_chief, target):
  worker_device = "/job:worker/task:%d/cpu:%d" % (task_index, 0)
  print("worker_deivce = %s" % worker_device)


  print "input tables: %s, output tables: %s" % (FLAGS.tables, FLAGS.outputs)
  with tf.device(worker_device):
    writer = tf.TableRecordWriter(FLAGS.outputs, slice_id=task_index)
    input_data = input_table_batch_fn(worker_count,
                                      task_index,
                                      global_config.input_instance[0]['table_name'],
                                      global_config.input_instance[0]['batch_size'],
                                      1,
                                      global_config.input_instance[0]['columns'],
                                      'input_table',
                                      True)
    print "input data batch_size:{}".format(global_config.input_instance[0]['batch_size'])
    print input_data
    fake_data = input_padding()
    fake_data.update(input_data)
    input_data = fake_data
    print input_data

  # assign global variables to ps nodes
  # with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index, cluster=cluster)):
  #   global_step = tf.Variable(0, name="global_step", trainable=False)
  with tf.device("/job:worker/task:%d" % task_index):
    global_step = tf.Variable(0, name="global_step", trainable=False)
    with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
      model = ModelFactory(FLAGS.model_name, global_config.model_instance[0], input_data)

    infer_types = set(FLAGS.infer_switch.split(','))
    res = []
    if 'query' in infer_types:
      query_rep = model.query_rep
      # query_rep_output = tf.reduce_join(tf.as_string(tf.transpose(query_rep)), 1, separator=',')
      # for TwoTowerMyBert no need to transpose
      query_rep_output = tf.reduce_join(tf.as_string(query_rep), 1, separator=',')
      query = input_data['query']
      # res.extend([query, query_rep_output])
      res.append(query)
    if 'doc' in infer_types:
      doc_rep = model.doc_rep
      # doc_rep_output = tf.reduce_join(tf.as_string(tf.transpose(doc_rep)), 1, separator=',')
      doc_rep_output = tf.reduce_join(tf.as_string(doc_rep), 1, separator=',')
      item_id = input_data['nid']
      res.extend([item_id, doc_rep_output])
    if 'score' in infer_types:
      score = model.predict
      res.append(score)
    if 'label' in infer_types:
      label = input_data['label']
      res.append(label)
    assert len(res) > 0
    write_to_odps = writer.write(range(0,len(res)), res)

  print("start training")

  config = tf.ConfigProto(
                          allow_soft_placement=True,
                          log_device_placement=False,
                          inter_op_parallelism_threads=128,
                          intra_op_parallelism_threads=128)
  config.gpu_options.allow_growth = True

  print FLAGS.saver_list
  saver_list = CreateSaverListFromConfig(FLAGS.saver_list)

  print_global_vairable()
  print_local_vairable()
  print time_stamp(), "Prepare enter session!"
  with tf.train.MonitoredTrainingSession(master=target,
                                         is_chief=True,
                                         checkpoint_dir=None,
                                         config=config,
                                         save_checkpoint_secs=None,
                                         save_summaries_steps=None,
                                         save_summaries_secs=None) as sess:

    print (time_stamp(), "Enter the session!")
    for s,c in saver_list:
      print ((time_stamp(), "restore from: %s" % (c)))
      s.restore(sess, c)
    print (time_stamp(), "Finish the restore!")

    step = 0
    while not sess.should_stop():
      step = step + 1
      _ = sess.run([write_to_odps])
      print time_stamp(), "%d batch writed" % step

  writer.close()


def main(unused_argv):
  print("job name = %s" % FLAGS.job_name)
  print("task index = %d" % FLAGS.task_index)
  is_chief = FLAGS.task_index == 0

  # construct the servers
  ps_spec = FLAGS.ps_hosts.split(",")
  worker_spec = FLAGS.worker_hosts.split(",")
  cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
  worker_count = len(worker_spec)
  ps_count = len(ps_spec)

  server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

  # join the ps server
  if FLAGS.job_name == "ps":
    server.join()

  # start the training
  #try:
  train(worker_count=worker_count, ps_count=ps_count, task_index=FLAGS.task_index, cluster=cluster, is_chief=is_chief,
          target=server.target)
  #except Exception, e:
  #  print("catch a exception: %s" % e.message)


if __name__ == "__main__":
  tf.app.run()
