# coding: utf-8

import tensorflow as tf
import numpy as np


# define the base input fn
def input_table_batch_fn(slice_count, slice_id, table_name, batch_size, num_epoches, schema_config, name_scope, allow_smaller_final_batch=False):
  print("slice_count:%d, slice_id:%d for table %s" % (slice_count, slice_id, table_name))
  print schema_config

  with tf.name_scope(name=name_scope):
    selected_col = ','.join([e['column_name'] for e in schema_config])
    file_queue = tf.train.string_input_producer([table_name], num_epochs=num_epoches)
    print selected_col
    reader = tf.TableRecordReader(slice_count=slice_count, slice_id=slice_id,csv_delimiter=',',
                                  selected_cols=selected_col,
                                  num_threads=18, capacity=batch_size*20)
    key, value = reader.read_up_to(file_queue, batch_size)
    print "value:{}".format(value)
    batch_res = tf.train.shuffle_batch([value], batch_size=batch_size, capacity=batch_size*20, enqueue_many=True,
                                       num_threads=8, min_after_dequeue=batch_size, allow_smaller_final_batch=allow_smaller_final_batch)
    print batch_res
    feature = tf.transpose(batch_res)
    print 'transpose feature {}'.format(feature)
    record_defaults = [[''] for _ in range(np.sum([e['length'] for e in schema_config]))]
    print 'record_defaults {}'.format(record_defaults)
    feature = tf.decode_csv(feature, record_defaults=record_defaults)
    print 'decode csv feature {}'.format(feature)

    res = {}
    length = len(schema_config)
    idx = 0
    start = 0
    for e in schema_config:
      datatype=e['dtype']
      length = e['length']
      name = e['name']
      if datatype == tf.string:
        b = feature[start:start + length]
      else:
        b = tf.string_to_number(feature[start:start + length], datatype)

      res[name]=tf.transpose(b, name=name)
      start = start + length

    res['batch_size']=batch_size
    return res



## feature column name, tensor_name, graph_name 应该是一样的