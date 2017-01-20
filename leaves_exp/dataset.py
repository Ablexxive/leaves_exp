from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import tensorflow as tf

from data_utils import load_data
class Dataset(object):

    def __init__(self, dataset_path, batch_size, samples_per_epoch):
    #batch_size would be used if we were using TF.records. Right now just building it piecemeal
        print("init print %s"%(dataset_path))
        self.dataset_dir = os.path.dirname(dataset_path)
        #self.dataset_path = dataset_path
        self.batch_size = batch_size
        # create an init flag
        self.examples_per_epoch = samples_per_epoch

    @property
    def steps_per_epoch(self):
    #change = examples / batch_size
        return self.batch_size * self.examples_per_epoch

    def get_input_fn(self, name,  num_epochs, shuffle):
    #a function that takes no argument and returns a tuple of (features, labels), where features is a dict of string key to Tensor and labels is a Tensor that's currently not used (and so can be None).
        #clarify w/ names and paths
        def input_fn():
            dataset_path = os.path.join(self.dataset_dir, name)
            dataset = load_data(dataset_path)
 
            # 1) wrap labels and feature in tf.Constant
            labels = tf.constant(dataset['labels'], name="labels")
            features = tf.constant(dataset['features'], name="features", dtype=tf.float32)

            # 2) Use 'tf.slice_input' to make single instances of a tf variables
            tensor_list = tf.train.slice_input_producer([labels, features],
                num_epochs=num_epochs, 
                shuffle=shuffle)             
            # 3) Use 'tf.shuffle_batch' to make a batch of single instance tf variables :) 
                #return outputs of suffle_batch
            label_tensor, feature_tensor = tf.train.shuffle_batch(
                tensor_list,
                batch_size=32,
                capacity=50000,
                min_after_dequeue=10000)           

            # dataset = load_data(dataset_path)
            return feature_tensor, label_tensor
        return input_fn
