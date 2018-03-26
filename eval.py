import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
import models
import math
import os
import sys
import importlib
import sklearn.preprocessing
import utils

# add configuration file
# Dictionary for model configuration
mdlParams = {}

# Import config
cfg = importlib.import_module('cfgs.'+sys.argv[1])
mdlParams.update(cfg.mdlParams)

# Get path where model is saved
mdlParams['saveDir'] = sys.argv[2]

# Trainset mean, subtracted from all volumes
if mdlParams['batch_func_type'] == 'getBatch3D':
    mdlParams['setMean'] = np.mean(mdlParams['data'][mdlParams['trainInd'],:,:,:])
    print("Setmean",mdlParams['setMean'])

# Scaler, scales targets to a range of 0-1
if mdlParams['scale_targets']:
    mdlParams['scaler'] = sklearn.preprocessing.MinMaxScaler().fit(mdlParams['targets'][mdlParams['trainInd'],:][:,mdlParams['tar_range'].astype(int)])

# Put all placeholders into one dictionary for feeding
placeholders = {}
# Values to feed during testing
feed_list_inference = {}
feed_list_inference['train_state'] = False

# Rebuild placeholders
input_size = [None]
input_size.extend(mdlParams['input_size'])
input_size.extend([1])
placeholders['X'] = tf.placeholder("float", input_size,name='X')
placeholders['Y'] = tf.placeholder("float", [None, len(mdlParams['tar_range'])],name='Y')
placeholders['train_state'] = tf.placeholder(tf.bool,name='train_state') #'train_state'

# Rebuild graph
model_function = models.getModel(mdlParams,placeholders)
prediction = model_function(placeholders['X'])

# Prepare getBatch function
getBatch = utils.getBatchFunction(mdlParams)

# Rebuild loss op
loss_op = tf.reduce_mean(tf.square(prediction-placeholders['Y']))

# Get moving average varibales
variable_averages = tf.train.ExponentialMovingAverage(mdlParams['moving_avg_var_decay'])
variables_to_restore = variable_averages.variables_to_restore(slim.get_model_variables())

# Get the saver to restore them
saver = tf.train.Saver(max_to_keep=0,var_list=variables_to_restore)

# Session config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Restore
with tf.Session(config=config) as sess:
    saver.restore(sess, tf.train.latest_checkpoint(mdlParams['saveDir'], 'checkpoint'))
    loss, mae_mean, mae_std, rmae_mean, rmae_std, acc = utils.getErrRegression(mdlParams, 'trainInd', sess, loss_op, prediction, getBatch, placeholders, feed_list_inference)
    print("Result on train set with moving average variables")
    print("-------------------------------------------------")
    print("Loss: ",loss," MAE: ",mae_mean,"+-",mae_std)
    print("Relative MAE: ",rmae_mean,"+-",rmae_std," aCC: ",acc,"\n")
    if 'valInd' in mdlParams:
        loss, mae_mean, mae_std, rmae_mean, rmae_std, acc = utils.getErrRegression(mdlParams, 'valInd', sess, loss_op, prediction, getBatch, placeholders, feed_list_inference)
        print("Result on valInd set with moving average variables")
        print("-------------------------------------------------")
        print("Loss: ",loss," MAE: ",mae_mean,"+-",mae_std)
        print("Relative MAE: ",rmae_mean,"+-",rmae_std," aCC: ",acc,"\n")
    if 'testInd' in mdlParams:
        loss, mae_mean, mae_std, rmae_mean, rmae_std, acc = utils.getErrRegression(mdlParams, 'testInd', sess, loss_op, prediction, getBatch, placeholders, feed_list_inference)
        print("Result on test set with moving average variables")
        print("-------------------------------------------------")
        print("Loss: ",loss," MAE: ",mae_mean,"+-",mae_std)
        print("Relative MAE: ",rmae_mean,"+-",rmae_std," aCC: ",acc)