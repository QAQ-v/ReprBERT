# coding=utf-8

import tensorflow as tf


def partitioner(ps_num, mem=8 * 1024 * 1024):
  partitioner = tf.min_max_variable_partitioner(
    max_partitions=ps_num*2,
    min_slice_size=mem,
  )
  return partitioner


class ModelBase(object):
  def __init__(self, config):
    self.FLAGS = config.FLAGS
    self.trainable = config.trainable
    #self.batch_size = config.batch_size
    self.batch_size = self.FLAGS.batch_size



    self.fixed_size_partitioner = tf.fixed_size_partitioner(self.FLAGS.partition_num, axis=0)
    self.fixed_size_partitioner = partitioner(self.FLAGS.partition_num)
    self.regularizer = tf.contrib.layers.l2_regularizer(self.FLAGS.weight_decay)


    if self.trainable:
      self.trainable_tensor =  tf.constant(True)
    else:
      self.trainable_tensor = tf.constant(False)

    self.resnet_infer_trainable = self.FLAGS.resnet_infer_trainable
    self.dssm_trainable = self.FLAGS.dssm_trainable


  def _add_training_summary(self):

    training_summary_name_tuple_list = [('loss', 'scalar'), ('learning_rate', 'scalar'), ('term1', 'scalar'), ('term2', 'scalar'), ('term3', 'scalar'),
                                        ('raw_gradient_global_normal', 'scalar'), ('clip_gradient_global_normal', 'scalar'), ('query_norm', 'scalar'), ('item_norm', 'scalar'),
                                        ('dis', 'histogram'), ('pos_distance','histogram'), ('pos_dis_mean', 'scalar'), ('train_dis_auc_1', 'scalar'), ('train_dis_auc_2', 'scalar'),
                                        ('left_margin','scalar'), ('right_margin','scalar'), ('title_attention_prob_entropy','scalar'), ('query_attention_prob_entropy', 'scalar'), ('within_right_margin','scalar'),
                                        ('pairwise_loss', 'scalar'), ('agreement_loss', 'scalar')]
    for k,v in training_summary_name_tuple_list:
      if k in self.__dict__:
        if v == 'scalar':
          if self.__dict__[k].shape.as_list() == []:
            tf.summary.scalar(k, self.__dict__[k])
          else:
            tf.summary.scalar(k, tf.reduce_mean(self.__dict__[k]))
        elif v == 'histogram':
          tf.summary.histogram(k, self.__dict__[k])
        else:
          raise ValueError("summary type error! %s, %s" % (k,v))

    for var in tf.global_variables():
      name_sp = var.name.split('/')
      if len(name_sp) >= 2 and name_sp[-2] == 'embedding':
        embedding_name = name_sp[-1]
        tf.summary.scalar(embedding_name, tf.reduce_mean(tf.norm(var, axis=1)))
        tf.summary.histogram(embedding_name, tf.norm(var, axis=1))

    # repretation summary
    query_mean, query_variance = tf.nn.moments(self.query_rep, axes=1)
    tf.summary.scalar("query_mean",tf.reduce_mean(query_mean))
    tf.summary.histogram("query_mean", query_mean)
    tf.summary.histogram("query_max", tf.reduce_max(self.query_rep, axis=1))
    tf.summary.histogram("query_min", tf.reduce_min(self.query_rep, axis=1))
    tf.summary.scalar("query_variance",tf.reduce_mean(query_variance))
    tf.summary.histogram("query_variance", query_variance)
    doc_mean, doc_variance = tf.nn.moments(self.doc_rep, axes=1)
    tf.summary.scalar("doc_mean",tf.reduce_mean(doc_mean))
    tf.summary.histogram("doc_mean", doc_mean)
    tf.summary.histogram("doc_max", tf.reduce_max(self.doc_rep, axis=1))
    tf.summary.histogram("doc_min", tf.reduce_min(self.doc_rep, axis=1))
    tf.summary.scalar("doc_variance", tf.reduce_mean(doc_variance))
    tf.summary.histogram("doc_variance", doc_variance)






  def model_fn_train(self, input_data):
    #self.batch_size = input_data["batch_size"]
    # maybe it have the same effort with create a variable named "global_steps"
    # but if create the variable here, maybe it have an specifc prefix
    self.global_step = tf.train.get_or_create_global_step()
    self.init_flag = tf.get_variable(name='init_flag', trainable=False, dtype=tf.int32, initializer=tf.constant(0))
    self.set_init_flag_op = tf.assign(self.init_flag, tf.constant(1))

    self._build_graph(input_data)
    if (self.trainable):
      print 'start to check batch_size'
      for k,v in input_data.items():
        print k, v
        if k != 'batch_size':
          assert self.batch_size == 0 or self.batch_size == v.shape.as_list()[0]
          self.batch_size = v.shape.as_list()[0]
      self._build_loss()
      self._build_dis_auc("train_dis_auc")
      self._build_train_op()
      self._add_training_summary()
    #else:
      #self._build_dis_auc("test_dis_auc")



    self.summaries = tf.summary.merge_all()

  def _build_dis_auc(self, name_scope):
    with tf.variable_scope(name_scope):
      _, self.train_dis_auc_1 = tf.contrib.metrics.streaming_auc(self.predict,
                                                                 self.label
                                                                 )
      # _, self.train_dis_auc_2 = tf.contrib.metrics.streaming_auc(tf.nn.sigmoid(self.predict),
      #                                                            self.label
      #                                                            )


  def _build_train_op(self):

    self.learning_rate = tf.maximum(tf.train.exponential_decay(self.FLAGS.learning_base,
                                                          self.global_step,
                                                          self.FLAGS.decay_step,
                                                          self.FLAGS.learning_rate_decay,
                                                          staircase=True),
                               1e-9)
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    if self.FLAGS.sync == True:
      optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=len(self.FLAGS.worker_hosts.split(",")) - 2,
                                   total_num_replicas=len(self.FLAGS.worker_hosts.split(",")), use_locking=True)
    self.optimizer = optimizer
    tvars = tf.trainable_variables()
    self.raw_gradient = tf.gradients(self.loss, tvars)
    grads, global_normal = tf.clip_by_global_norm(self.raw_gradient, self.FLAGS.max_grad_norm)
    self.raw_gradient_global_normal = global_normal
    self.clip_gradient_global_normal = tf.global_norm(grads)
    apply_gradient_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
    variable_averages = tf.train.ExponentialMovingAverage(self.FLAGS.moving_average_decay, self.global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      self.train_op = tf.no_op(name='train')


  def _build_pai_soar_train_op(self):
    self.learning_rate = tf.maximum(tf.train.exponential_decay(self.FLAGS.learning_base,
                                                          self.global_step,
                                                          self.FLAGS.decay_step,
                                                          self.FLAGS.learning_rate_decay,
                                                          staircase=True),
                               1e-9)
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    optimizer = pai.ReplicatedVarsOptimizer(optimizer)  # <- Optimizer封装
    self.optimizer = optimizer
    self.raw_gradient = 0
    self.raw_gradient_global_normal = 0
    self.clip_gradient_global_normal = 0
    def loss_fn():
      return self.loss
    loss = optimizer.compute_loss(loss_fn)
    self.train_op = optimizer.minimize(loss)

  def _build_graph(self, input_data):
    pass

  def _build_loss(self):
    pass

