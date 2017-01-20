from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

def basic_model(x):
    # No activation function
    print("Running Model 1")
    print(x.get_shape())
    #So the model is expecing a Tensor of 1 dimenstion (since I'm not one-hotting input)
    W = tf.Variable(tf.zeros([192, 99]))#, dtype=tf.float64))
    b = tf.Variable(tf.zeros([99]))#, dtype=tf.float64))
    y = tf.matmul(x, W) + b
    print(W)
    print(b)
    print(y)
    return y

def model_fn(features, labels, params, mode, scope=None):
    #Following Pieces:
    #1. Configure model via TensorFlow operations
    #2. Define loss function for training/eval
    #3. Define training operation/optimizer
    #4. Generate Predictions
    initializer = tf.random_normal_initializer(stddev=0.1) #TODO: Pull He from model_utils  
    with tf.variable_scope(scope, 'BasicNetwork', initializer=initializer): 
        print(features)
        logits = basic_model(features)
        output = tf.nn.softmax(logits)
        
        loss = get_loss(output, labels, mode) 
        train_op = get_train_op(loss, params, mode)

        prediction = tf.argmax(output, 1) 
        
    return prediction, loss, train_op

def get_loss(output, labels, mode):
    if mode == tf.contrib.learn.ModeKeys.INFER:
        return None
    # The sparse softmax here lets TF know that you'll have categorical flags and it will one-hot them automatically
    print("Sparse softmax input:")
    print(labels)
    print("Shape of outputs:")
    print(output.get_shape())
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output))

def get_train_op(loss, params, mode):
    if mode != tf.contrib.learn.ModeKeys.TRAIN:
        return None
    
    learning_rate_init = params['learning_rate']
    learning_rate_decay_rate = params['learning_rate_decay_rate']
    learning_rate_decay_steps = params['learning_rate_decay_steps']

    global_step = tf.contrib.framework.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
        learning_rate=learning_rate_init,
        decay_steps=learning_rate_decay_steps,
        decay_rate=learning_rate_decay_rate,
        global_step=global_step,
        staircase=True)

    # Allows 'leanring_rate' to be shown on TensorBoard
    tf.contrib.layers.summarize_tensor(learning_rate, tag='learning_rate')

    train_op = tf.contrib.layers.optimize_loss(loss,
        global_step=global_step,
        learning_rate=learning_rate,
        optimizer='Adam')
    #clip-gradients - useful for RNNs since the graidents can multiply out

    return train_op
