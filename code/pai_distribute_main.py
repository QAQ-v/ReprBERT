# coding: utf-8

# In[1]:

import os
import numpy as np
import random
from common.util.util import *
from common.util.config_util import *
from common.util.init_util import *
from common.input.tf_input import *
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.python.client import timeline
from evaluation.base import *
from model.factory import *
import sys
reload(sys)
sys.setdefaultencoding('utf8')

np.set_printoptions(threshold=sys.maxsize)

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


def train(worker_count, ps_count, task_index, cluster, is_chief, target, is_train_worker):
  worker_device = "/job:worker/task:%d" % (task_index)
  print("worker_deivce = %s" % worker_device)
  g_2 = tf.Graph()

  # assign io related variables and ops to local worker device
  with tf.device(worker_device):
    input = []
    evals = []
    for i in range(len(global_config.input_instance)):
      if global_config.input_instance[i]['train'] == is_train_worker :
        if is_train_worker == True:
          input_temp = input_table_batch_fn(global_config.input_instance[i]['slice_count'],
                                       global_config.input_instance[i]['slice_id'],
                                       global_config.input_instance[i]['table_name'],
                                       global_config.input_instance[i]['batch_size'],
                                       global_config.input_instance[i]['epoch'],
                                       global_config.input_instance[i]['columns'],
                                       'input_table',
                                       False)
          input.append(input_temp)
        else:
          for e in global_config.input_instance[i]['evaluation']:
            evaluation = CreateEvaluation(e,
                                          FLAGS.model_name,
                                          global_config.input_instance[i])
            evaluation._build_input()
            evals.append(evaluation)
    if is_train_worker:
      assert len(input) == 1
    else:
      assert len(input) == 0

    if is_train_worker == False:
      with g_2.as_default():
        with tf.name_scope("evaluation") as scope:
          for e in evals:
            e._build_summary()

  # assign global variables to ps nodes
  with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index, cluster=cluster)):
    global_step = tf.Variable(0, name="global_step", trainable=False)
    if is_train_worker == True:
      with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
        model = ModelFactory(FLAGS.model_name, global_config.model_instance[0], input[0])
    else:
      reuse=tf.AUTO_REUSE
      for e in evals:
        e._build_eval_op(reuse, global_config.model_instance[1])
        reuse=True
      model = evals[0].model

  print("start training")

  config = tf.ConfigProto(#device_count={"CPU": 4},  # limit to num_cpu_core CPU usage
                          allow_soft_placement=True,
                          log_device_placement=False,
                          inter_op_parallelism_threads=64,
                          intra_op_parallelism_threads=64)
  config.gpu_options.allow_growth = True
  # tracing_hook = tf.train.TracingHook(
  #   os.path.join(FLAGS.checkpointBaseDir+FLAGS.ModelTag+'/train-%s-%s.timeline' % \
  #                       (FLAGS.job_name, task_index)),
  #   10020,
  #   min_vtrace_level=1)
  train_steps = FLAGS.data_size // FLAGS.batch_size * FLAGS.num_epoches
  print("train_steps = ", train_steps)
  if FLAGS.sync == True and is_train_worker == True:
    hooks = [tf.train.StopAtStepHook(last_step=train_steps), model.optimizer.make_session_run_hook(is_chief)]
  else:
    hooks = [tf.train.StopAtStepHook(last_step=train_steps)]

  print("Prepare enter Session!")
  stream_vars = [i for i in tf.local_variables() if 'my_streaming' in i.name]
  reset_op = [tf.initialize_variables(stream_vars)]

  print FLAGS.saver_list
  saver_list = CreateSaverListFromConfig(FLAGS.saver_list)
  print saver_list

  print_global_vairable()
  print_local_vairable()

  print("checkpoint_dir =", FLAGS.checkpointBaseDir+FLAGS.ModelTag)
  scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=1000))
  with tf.train.MonitoredTrainingSession(master=target, is_chief=is_chief,
                                         checkpoint_dir=FLAGS.checkpointBaseDir+FLAGS.ModelTag,
                                         hooks=hooks,
                                         config=config,
                                         save_checkpoint_secs=FLAGS.save_checkpoint_secs if FLAGS.save_checkpoint_secs != 0 else None,
                                         save_summaries_steps=100,
                                         scaffold=scaffold) as mon_sess:

    print (time_stamp(), "Enter the session!")
    if (FLAGS.initw2v_worker_index == task_index):
      for s,c in saver_list:
        print ((time_stamp(), "restore from: %s" % (c)))
        s.restore(mon_sess, c)
      print (time_stamp(), "Finish the restore!")

      mon_sess.run(model.set_init_flag_op)
      print (time_stamp(), "Finish the init!")
      time.sleep(10)
    else:
      while (True):
        print (time_stamp(), "run the init flag!")
        init_flag_value = mon_sess.run(model.init_flag)
        if init_flag_value == 1:
          break;
        print (time_stamp(), "wait for the init!")
        time.sleep(30)

    print "prepare to run!"
    step = 0
    while not mon_sess.should_stop():
      try:
        step = step + 1

        if is_train_worker == False:
          mon_sess.run(reset_op)
          a = mon_sess.run([model.global_step])[0]
          for e in evals:
            e._run_session(session=mon_sess, step =step, global_step=a)
          with g_2.as_default(),tf.Session() as sess:
            summary_writer_monitoredtrainingsession = SummaryWriterCache.get(FLAGS.checkpointMetricDir+FLAGS.ModelTag)
            for e in evals:
              e._feed_summary(sess, summary_writer_monitoredtrainingsession, a)
          time.sleep(60)

        else:
          if step % 100 == 0:
            #print "run train with log"
            if 'query_attention_prob' in model.__dict__ and 'title_attention_prob' in model.__dict__:
              run_meta_data = tf.RunMetadata()
              a, b, c, d, e, _ = mon_sess.run([model.global_step, model.learning_rate, model.loss,model.query_attention_prob, model.title_attention_prob, model.train_op])
              print "%s Training Step %d / %d, learning rate: %f, loss : %f" % (time_stamp(), step, a, b, c)
              r = random.randint(0, global_config.input_instance[0]['batch_size'] - 1)
              print r
              print str(d[r])
              print str(e[r])
              tl = timeline.Timeline(run_meta_data.step_stats)
              content = tl.generate_chrome_trace_format()
              tl_path = os.path.join(FLAGS.checkpointBaseDir + FLAGS.ModelTag + '/timeline/train-%s-%s.timeline' % (
              FLAGS.job_name, task_index))
              with tf.gfile.Open(tl_path, 'w') as f:
                f.write(content)
              tf.logging.info("success to generate the timeline file %s" % (tl_path))
            else:
              a, b, c, _ = mon_sess.run(
                [model.global_step, model.learning_rate, model.loss, model.train_op])
              print "%s Training Step %d / %d, learning rate: %f, loss : %f" % (time_stamp(), step, a, b, c)



          else:
            a, b, _, = mon_sess.run([model.global_step, model.learning_rate, model.train_op])
      except tf.errors.OutOfRangeError:
        print "Finished by tf.errors.OutOfRangeError"
        break


def main(unused_argv):
  print("job name = %s" % FLAGS.job_name)
  print("task index = %d" % FLAGS.task_index)
  is_chief = FLAGS.task_index == 0
  print "FLAGS ="
  print FLAGS.checkpointDir
  print FLAGS

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

  print "model name: %s, input_name: %s, vocab_size: %d, initw2v: %d" % \
        (FLAGS.model_name, FLAGS.input_name, FLAGS.vocab_size, FLAGS.initw2v)
  train(worker_count=worker_count, ps_count=ps_count, task_index=FLAGS.task_index, cluster=cluster, is_chief=is_chief,
          target=server.target, is_train_worker=FLAGS.task_index != worker_count -1)

if __name__ == "__main__":
  tf.app.run()