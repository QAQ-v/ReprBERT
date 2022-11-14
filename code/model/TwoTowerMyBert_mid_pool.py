# coding=utf-8

from base import *
from common.module import basic_component
import common.module.bert.modeling as bert
class Model(ModelBase):

  def _build_graph(self, input_data):

    #batch_size = self.config.batch_size if self.trainable else self.config.batch_size_test
    # batch_size = input_data['batch_size']
    query_length = self.FLAGS.query_length
    doc_length = self.FLAGS.doc_length
    weight_decay = self.FLAGS.weight_decay
    self.margin = self.FLAGS.margin

    filter_num = 80
    fc_size = 256
    k = 1

    print "zixuan input: ", input_data

    output_embedding_dim = 128

    self.bert_config = bert.BertConfig(vocab_size=self.FLAGS.bert_vocab_size,
                                       hidden_size=self.FLAGS.bert_hidden_size,
                                       num_hidden_layers=self.FLAGS.bert_num_hidden_layers,
                                       num_attention_heads=self.FLAGS.bert_num_attention_heads,
                                       intermediate_size=self.FLAGS.bert_intermediate_size,
                                       hidden_act=self.FLAGS.bert_hidden_act,
                                       hidden_dropout_prob=self.FLAGS.bert_hidden_dropout_prob,
                                       attention_probs_dropout_prob=self.FLAGS.bert_attention_probs_dropout_prob,
                                       max_position_embeddings=self.FLAGS.bert_max_position_embeddings,
                                       type_vocab_size=self.FLAGS.bert_type_vocab_size,
                                       initializer_range=self.FLAGS.bert_initializer_range)
    with tf.variable_scope("bert_layer", reuse=tf.AUTO_REUSE, partitioner=self.fixed_size_partitioner):
      query_bert_model = bert.BertModel(config=self.bert_config,
                                        is_training=self.trainable,
                                        input_ids = input_data['ids'],
                                        input_mask=input_data['mask'],
                                        token_type_ids=input_data['seg_ids'],
                                        use_one_hot_embeddings=False,
                                        scope="bert")
      # [b, n, d] , mask: [b, n]
      mask = tf.cast(tf.expand_dims(input_data['mask'], dim=2), dtype=tf.float32)
      query_bert_output = query_bert_model.get_sequence_output() #* tf.cast(tf.expand_dims(input_data['mask'], dim=2), dtype=tf.float32) #basic_component.get_bert_pooled_output(query_bert_model, input_data['mask'])
      print "query_bert_output: ", query_bert_output
      print "mask: ", mask
      # query_bert_output = tf.Print(query_bert_output, [mask, query_bert_output, query_bert_output*mask], message="query_bert_output: ",
      #   first_n=1, summarize=1000)
      query_bert_output = query_bert_output * mask
      query_bert_mid_outputs = query_bert_model.get_all_encoder_layers()

      title_bert_model = bert.BertModel(config=self.bert_config,
                                        is_training=self.trainable,
                                        input_ids = input_data['ids1'],
                                        input_mask=input_data['mask1'],
                                        token_type_ids=input_data['seg_ids1'],
                                        use_one_hot_embeddings=False,
                                        scope="bert")
      mask1 = tf.cast(tf.expand_dims(input_data['mask1'], dim=2), dtype=tf.float32)
      title_bert_output = title_bert_model.get_sequence_output() #* tf.cast(tf.expand_dims(input_data['mask1'], dim=2), dtype=tf.float32) # basic_component.get_bert_pooled_output(title_bert_model, input_data['mask1'])
      print "title_bert_output: ", title_bert_output
      print "mask1: ", mask1
      # title_bert_output = tf.Print(title_bert_output, [mask, title_bert_output, title_bert_output*mask], message="title_bert_output: ",
      #   first_n=1, summarize=1000)
      title_bert_output = title_bert_output * mask1
      title_bert_mid_outputs = title_bert_model.get_all_encoder_layers()



    with tf.variable_scope('text_fc_layer', partitioner=self.fixed_size_partitioner):
      querylen = query_bert_output.shape[1].value
      titlelen = title_bert_output.shape[1].value
      batch_size = tf.shape(title_bert_output)[0] # here need to check dynamic dimension


      query_pool_weight = tf.get_variable(
        "query_pool_weight", [1, querylen],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

      title_pool_weight = tf.get_variable(
        "title_pool_weight", [1, titlelen],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

      # output_bias = tf.get_variable(
      #     "output_bias", [num_labels], initializer=tf.zeros_initializer())
      # weight average pooling
      # query_shape = query_bert_output.shape.as_list()
      # title_shape = title_bert_output.shape.as_list()

      # query_bert_output = tf.reshape(query_bert_output, [querylen, batch_size*self.FLAGS.bert_hidden_size])
      # title_bert_output = tf.reshape(title_bert_output, [titlelen, batch_size*self.FLAGS.bert_hidden_size])
      query_pool_weight = tf.tile(tf.expand_dims(query_pool_weight, axis=0), [batch_size, 1, 1])
      title_pool_weight = tf.tile(tf.expand_dims(title_pool_weight, axis=0), [batch_size, 1, 1])
      print("query_pool_weight =", query_pool_weight)
      print("title_pool_weight =", title_pool_weight)

      query_pool_output = tf.matmul(query_pool_weight, query_bert_output)
      title_pool_output = tf.matmul(title_pool_weight, title_bert_output)

      # query_pool_output = tf.reshape(query_pool_output, [batch_size, 1, self.FLAGS.bert_hidden_size])
      # title_pool_output = tf.reshape(title_pool_output, [batch_size, 1, self.FLAGS.bert_hidden_size])
      

      print("query_pool_output =", query_pool_output)
      print("title_pool_output =", title_pool_output)
      self.query_rep = query_pool_output = tf.squeeze(query_pool_output, axis=1, name='query_final_rep')
      self.doc_rep = title_pool_output = tf.squeeze(title_pool_output, axis=1, name='title_final_rep')
      print("query_pool_output_squeeze =", query_pool_output)
      print("title_pool_output_squeeze =", title_pool_output)

      # fusion_bert_output = tf.math.maximum(title_pool_output, query_pool_output) for 1.12
      fusion_bert_output = tf.maximum(title_pool_output, query_pool_output)

      with tf.name_scope('mid_interaction'):
        fusion_mid_outputs = []
        fusion_mid_query_outputs, fusion_mid_title_outputs = [], []
        query_mid_pool_weights, title_mid_pool_weights = [], []

        for i in range(self.FLAGS.bert_num_hidden_layers-1):
          query_mid_pool_weight = tf.get_variable("query_mid_pool_weight_%s" % str(i), [1, querylen],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
          title_mid_pool_weight = tf.get_variable("title_mid_pool_weight_%s" % str(i), [1, titlelen],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

          query_mid_pool_weight = tf.tile(tf.expand_dims(query_mid_pool_weight, axis=0), [batch_size, 1, 1])
          title_mid_pool_weight = tf.tile(tf.expand_dims(title_mid_pool_weight, axis=0), [batch_size, 1, 1])

          query_mid_pool_weights.append(query_mid_pool_weight)
          title_mid_pool_weights.append(title_mid_pool_weight)

          fusion_mid_query_outputs.append(tf.squeeze(
            tf.matmul(query_mid_pool_weights[i], query_bert_mid_outputs[i]), axis=1, name='query_mid_rep_%s'%str(i)))
          fusion_mid_title_outputs.append(tf.squeeze(
            tf.matmul(title_mid_pool_weights[i], title_bert_mid_outputs[i]), axis=1, name='title_mid_rep_%s'%str(i)))

          fusion_mid_outputs.append(tf.maximum(fusion_mid_query_outputs[i], fusion_mid_title_outputs[i]))

        self.fusion_mid_outputs = fusion_mid_outputs = tf.concat(fusion_mid_outputs, axis=-1)

      fusion_final_weight = tf.get_variable(
        "fusion_final_weight", [self.FLAGS.bert_hidden_size * self.FLAGS.bert_num_hidden_layers, self.FLAGS.bert_hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
      # fusion_final_weight = tf.tile(tf.expand_dims(fusion_final_weight, axis=0), [batch_size, 1, 1])

      self.fusion_final_output = fusion_final_output = tf.matmul(
        tf.concat([fusion_mid_outputs, fusion_bert_output], axis=-1), fusion_final_weight)

      # self.fusion_final_output = fusion_final_output = tf.squeeze(fusion_final_output, axis=1)





      # if self.trainable:   
        # I.e., 0.1 dropout
        # fusion_bert_output = tf.nn.dropout(fusion_bert_output, keep_prob=0.9)

      fusion_bert_output_1 = basic_component._add_fc_layer_shared_by_name(fusion_final_output,
                                                                   self.FLAGS.bert_hidden_size,
                                                                   self.FLAGS.bert_hidden_size,
                                                                   "dense1",
                                                                     activation_function=tf.nn.relu) + fusion_bert_output



      fusion_bert_output_2 = basic_component._add_fc_layer_shared_by_name(fusion_bert_output_1,
                                                                   self.FLAGS.bert_hidden_size,
                                                                   self.FLAGS.bert_hidden_size,
                                                                   "dense2",activation_function=tf.nn.relu) + fusion_bert_output_1

      self.final_fusion_ouput = basic_component._add_fc_layer_shared_by_name(fusion_bert_output_2,
                                                                     self.FLAGS.bert_hidden_size,
                                                                     2,
                                                                     "dense3") 
    with tf.name_scope('loss'):

      if self.trainable:
        self.teacher_logits = tf.reshape(input_data['teacher_logits'], [input_data['batch_size'], 2])
        # self.label = tf.math.argmax(self.teacher_logits, axis=-1) # for 1.12
        self.label = tf.argmax(self.teacher_logits, axis=-1) # 


      self.probabilities = tf.nn.softmax(self.final_fusion_ouput, axis=-1, name='out_logits')
      self.log_probs = tf.nn.log_softmax(self.final_fusion_ouput, axis=-1)
      self.pos_distance = self.predict = self.probabilities[:, 1]

      

  def _build_loss(self):
    with tf.name_scope('loss'):
      per_example_loss = -tf.reduce_sum(self.teacher_logits * self.log_probs, axis=-1)
      self.loss = tf.reduce_mean(per_example_loss)
