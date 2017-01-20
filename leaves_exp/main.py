from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import time

import random
import numpy as np
import tensorflow as tf

from sacred import Experiment

from model import model_fn
from dataset import Dataset
from data_utils import load_data

#tf.logging.set_verbosity(tf.logging.INFO)
ex = Experiment('sacred_test')
LOGS_DIRECTORY = os.environ.get('LOGS_DIRECTORY', 'logs/')
DATA_DIRECTORY = os.environ.get('DATA_DIRECTORY', './datasets/')


@ex.config
def config():
    #move flags here
    seed = 42
    model_dir = 'logs/'
    dataset = './datasets/train.csv'
    num_epochs = 200
    learning_rate = 1e-3
    learning_rate_decay_rate = 0.1
    learning_rate_decay_steps = 6000
    early_stopping_rounds = 10

    flags = {
        'seed':seed,
        'model_dir':model_dir,
        'dataset':dataset,
        'num_epochs':200,
        'learning_rate':learning_rate,
        'learning_rate_decay_rate':learning_rate_decay_rate,
        'learning_rate_decay_steps':6000,
        'early_stopping_rounds':10,
    }

#FLAGS = tf.app.flags.FLAGS

#Create Flags here as needed
#tf.app.flags.DEFINE_integer/float/string/boolean('flag_name', default, 'info')
#tf.app.flags.DEFINE_integer('seed', 42, 'Random seed.')
#tf.app.flags.DEFINE_string('model_dir', 'logs/', 'Model directory.')
#tf.app.flags.DEFINE_string('dataset', './datasets/train.csv', 'Path of dataset.')

#tf.app.flags.DEFINE_integer('num_epochs', 200, 'Number of training epochs.')
#tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
#tf.app.flags.DEFINE_float('learning_rate_decay_rate', 0.1, 'Learning rate decay rate.')
#tf.app.flags.DEFINE_float('learning_rate_decay_steps', 6000, 'Learning rate decay steps.')
#tf.app.flags.DEFINE_integer('early_stopping_rounds', 10, 'Number of epochs before early stopping.')

#@ex.automain
#def run(flags):
@ex.automain
def main(flags):
    random.seed(flags["seed"])
    np.random.seed(flags["seed"])

    dataset = Dataset(DATA_DIRECTORY, 32, 1000)
    # both train/eval input_fn's are built off of the training data just so we have the pipleine tested.
    train_input_fn = dataset.get_input_fn('train.csv',
        num_epochs=flags["num_epochs"],
        shuffle=True)
    eval_input_fn = dataset.get_input_fn('train.csv',
        num_epochs=1,
        shuffle=False)

    #create session and run to test that features/labeles are read in properly  
    features, labels = train_input_fn() #this line is for testing, remove
    print("Features!")
    print(features)
    print("Labels!")
    print(labels) 
    params = {
        'learning_rate': flags["learning_rate"],
        'learning_rate_decay_rate': flags["learning_rate_decay_rate"],
        'learning_rate_decay_steps': flags["learning_rate_decay_steps"],
    } #Non learned parameters for the model such as learning rate
    
    train_monitors = [
        # Run a validation pass every epoch - this gets passed to experiment
        #runs during training and *not* validatio (monitors vs metrics)
        # Early stopping checks for 
        tf.contrib.learn.monitors.ValidationMonitor(
            input_fn=train_input_fn,
            every_n_steps=dataset.steps_per_epoch,
            early_stopping_rounds=flags["early_stopping_rounds"] * dataset.steps_per_epoch,
            early_stopping_metric='loss',
            early_stopping_metric_minimize=True)
    ]
 
    #eval_metrics = {}-apparently ValidationMonitor collects loss + accuracy by default, lets see
    # are logged differently - also runs every time a checkpoint is saved
    eval_metrics = {
            "accuracy": tf.contrib.learn.metric_spec.MetricSpec(tf.contrib.metrics.streaming_accuracy)
    } 
    
    #Config file for how the model is saved/logged
    config = tf.contrib.learn.RunConfig(
        tf_random_seed=flags["seed"],
        save_summary_steps=120,#how often the model is evaluated?
        save_checkpoints_secs=600,#how often the model is saved
        keep_checkpoint_max=5,
        keep_checkpoint_every_n_hours=1,
        log_device_placement=True) #?

    dataset_name = os.path.splitext(os.path.basename(flags["dataset"]))[0]
    timestamp = int(time.time())
    model_dir = os.path.join(flags["model_dir"], dataset_name, str(timestamp))
    estimator = tf.contrib.learn.Estimator(
        model_dir=model_dir,
        model_fn=model_fn,
        config=config,
        params=params)

    experiment = tf.contrib.learn.Experiment(
        estimator,
        train_input_fn,
        eval_input_fn,
        train_steps=None,
        eval_steps=None,
        eval_metrics=eval_metrics,
        train_monitors=train_monitors,
        local_eval_frequency=1)

    experiment.train_and_evaluate()

