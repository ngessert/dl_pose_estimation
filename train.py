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
import time
import sklearn.preprocessing
import utils

# add configuration file
# Dictionary for model configuration
mdlParams = {}

# Import config
cfg = importlib.import_module('cfgs.'+sys.argv[1])
mdlParams.update(cfg.mdlParams)

# Check if there is a validation set, if not, evaluate train error instead
if 'valInd' in mdlParams:
    eval_set = 'valInd'
    print("Evaluating on validation set during training.")
else:
    eval_set = 'trainInd'
    print("No validation set, evaluating on training set during training.")

# Trainset mean, subtracted from all volumes
if mdlParams['batch_func_type'] == 'getBatch3D':
    mdlParams['setMean'] = np.mean(mdlParams['data'][mdlParams['trainInd'],:,:,:])
    print("Setmean",mdlParams['setMean'])

# Scaler, scales targets to a range of 0-1
if mdlParams['scale_targets']:
    mdlParams['scaler'] = sklearn.preprocessing.MinMaxScaler().fit(mdlParams['targets'][mdlParams['trainInd'],:][:,mdlParams['tar_range'].astype(int)])

# Put all placeholders into one dictionary for feeding
placeholders = {}
# Values to feed during training
feed_list = {}
# Values to feed during testing
feed_list_inference = {}

# Define model input
input_size = [None]
input_size.extend(mdlParams['input_size'])
input_size.extend([1])
placeholders['X'] = tf.placeholder("float", input_size,name='X')
placeholders['Y'] = tf.placeholder("float", [None, len(mdlParams['tar_range'])],name='Y')
placeholders['train_state'] = tf.placeholder(tf.bool,name='train_state') 
# Value to feed for training/testing
feed_list['train_state'] = True
feed_list_inference['train_state'] = False

# Build graph, Inception3D, ResNetA3D, ResNetB3D, ResNeXt3D
model_function = models.getModel(mdlParams,placeholders)
prediction = model_function(placeholders['X'])

# Prepare getBatch function
getBatch = utils.getBatchFunction(mdlParams)

# Summaries
summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

# Def global step
global_step = tf.Variable(0, name='global_step', trainable=False)

# Add variable summaries
for var in slim.get_model_variables():
    summaries.append(tf.summary.histogram(var.op.name, var))

# Moving average variables
moving_average_variables = slim.get_model_variables()
variable_averages = tf.train.ExponentialMovingAverage(mdlParams['moving_avg_var_decay'], global_step)

# Define loss/training
with tf.name_scope('Optimization'):
    # Get all update ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    loss_op = tf.reduce_mean(tf.square(prediction-placeholders['Y']))
    # Changeable learning rate
    lr = tf.get_variable("learning_rate", [],trainable=False,initializer=init_ops.constant_initializer(mdlParams['learning_rate']))
    optimizer = tf.train.AdamOptimizer(lr,0.9,0.999,0.000001)
    grads = optimizer.compute_gradients(loss_op,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    # Add gradient summaries
    global_grad_norm = clip_ops.global_norm(list(zip(*grads))[0])
    # add summary for gradient norm
    summaries.append(tf.summary.scalar("global_norm/gradient_norm",global_grad_norm))
    # And for all gradients
    for grad, var in grads:
       summaries.append(tf.summary.scalar(var.op.name + '/gradient_norm',clip_ops.global_norm([grad])))
    # Apply gradients with control dependencies
    update_ops.append(variable_averages.apply(moving_average_variables))
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grads, global_step = global_step,name='train_op')

# Merge summaries
all_summaries = tf.summary.merge(summaries)

# Saver
saver = tf.train.Saver(max_to_keep=0)

# Check if there is something to load
load_old = 0
if os.path.isdir(mdlParams['saveDir']):
    # Check if a real checkpoint is in there (more than 4 files)
    if len([name for name in os.listdir(mdlParams['saveDir'])]) > 4:
        load_old = 1
        print("Loading old model")
    else:
        # Delete whatever is in there
        filelist = [os.remove(mdlParams['saveDir'] +'/'+f) for f in os.listdir(mdlParams['saveDir'])]
else:
    os.mkdir(mdlParams['saveDir'])

# Initialize the variables (i.e. assign their default value)
if not load_old:
    init = tf.global_variables_initializer()

# Num batches
numBatchesTrain = int(math.ceil(len(mdlParams['trainInd'])/mdlParams['batchSize']))
print("Train batches",numBatchesTrain)

# Session config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Track metrics for lowering LR
mdlParams['lastBestInd'] = -1
mdlParams['valBest'] = 1000
mdlParams['lastLRUpdate'] = 0

# Start training
with tf.Session(config=config) as sess:
    # Run the initializer or initialize old model
    if load_old:
        saver.restore(sess, tf.train.latest_checkpoint(mdlParams['saveDir'], 'checkpoint'))
    else:
        sess.run(init)
    # Initialize summaries writer
    sum_writer = tf.summary.FileWriter(mdlParams['saveDir'], sess.graph)
    # Run training
    start_time = time.time()
    print("Start training...")
    for step in range(1, mdlParams['training_steps']+1):
        # Shuffle inds in every epoch
        np.random.shuffle(mdlParams['trainInd'])
        for j in range(numBatchesTrain):
            feed_list['X'], feed_list['Y'] = getBatch(mdlParams,'trainInd',j)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={placeholders[p]: feed_list[p] for p in placeholders})
        if step % mdlParams['display_step'] == 0 or step == 1:
            # Duration so far
            duration = time.time() - start_time
            # Update summaries, take last batch
            feed_list_inference['X'] = feed_list['X']
            feed_list_inference['Y'] = feed_list['Y']
            summary_str = sess.run(all_summaries, feed_dict={placeholders[p]: feed_list_inference[p] for p in placeholders})
            sum_writer.add_summary(summary_str, step)
            # Calculate evaluation metrics
            loss, mae_mean, mae_std, rmae_mean, rmae_std, acc = utils.getErrRegression(mdlParams, eval_set, sess, loss_op, prediction, getBatch, placeholders, feed_list_inference)
            # Track all metrics as summaries
            sum_mae_mean = tf.Summary(value=[tf.Summary.Value(tag='MAE_mean', simple_value = mae_mean)])
            sum_writer.add_summary(sum_mae_mean, step)
            sum_mae_std = tf.Summary(value=[tf.Summary.Value(tag='MAE_std', simple_value = mae_std)])
            sum_writer.add_summary(sum_mae_std, step)
            sum_rmae_mean = tf.Summary(value=[tf.Summary.Value(tag='rMAE_mean', simple_value = rmae_mean)])
            sum_writer.add_summary(sum_rmae_mean, step)
            sum_rmae_std = tf.Summary(value=[tf.Summary.Value(tag='rMAE_std', simple_value = rmae_std)])
            sum_writer.add_summary(sum_rmae_std, step)
            sum_acc= tf.Summary(value=[tf.Summary.Value(tag='aCC', simple_value = acc)])
            sum_writer.add_summary(sum_acc, step)
            # Check if we have a new best value
            if mae_mean < mdlParams['valBest']:
                mdlParams['valBest'] = mae_mean
                oldBestInd = mdlParams['lastBestInd']
                mdlParams['lastBestInd'] = step
                # Delte previously best model
                if os.path.isfile(mdlParams['saveDir'] + '/checkpoint-' + str(oldBestInd) + '.index'):
                    os.remove(mdlParams['saveDir'] + '/checkpoint-' + str(oldBestInd) + '.data-00000-of-00001')
                    os.remove(mdlParams['saveDir'] + '/checkpoint-' + str(oldBestInd) + '.index')
                    os.remove(mdlParams['saveDir'] + '/checkpoint-' + str(oldBestInd) + '.meta')
                # Save currently best model
                saver.save(sess, mdlParams['saveDir'] + '/checkpoint', global_step=step)
            # Print
            print("\n")
            print('Epoch: %d/%d (%d h %d m %d s)' % (step,mdlParams['training_steps'], int(duration/3600), int(np.mod(duration,3600)/60), int(np.mod(np.mod(duration,3600),60))) + time.strftime("%d.%m.-%H:%M:%S", time.localtime()))
            print("Loss on ",eval_set,"set: ",loss," MAE: ",mae_mean,"+-",mae_std," (best MAE: ",mdlParams['valBest']," at Epoch ",mdlParams['lastBestInd'],")")
            print("Relative MAE: ",rmae_mean,"+-",rmae_std," aCC: ",acc)
            # Potentially peek at test error
            if mdlParams['peak_at_testerr']:
                loss, mae_mean, mae_std, rmae_mean, rmae_std, acc = utils.getErrRegression(mdlParams, 'testInd', sess, loss_op, prediction, getBatch, placeholders, feed_list_inference)
                print("\n")
                print("Test loss: ",loss," MAE: ",mae_mean,"+-",mae_std)
                print("Relative MAE: ",rmae_mean,"+-",rmae_std," aCC: ",acc)
            # Flush
            sys.stdout.flush()
            sys.stderr.flush()
        # maybe adjust LR
        if eval_set == 'valInd':
            cond = (step-mdlParams['lastBestInd']) >= mdlParams['lowerLRAfter'] and (step-mdlParams['lastLRUpdate']) >= mdlParams['lowerLRAfter']
        else:
            cond = (mdlParams['lowerLRat'] + mdlParams['lowerLRAfter']*mdlParams['lastLRUpdate']) < step
        if cond:
            oldLR = sess.run(lr)
            print("Old Learning Rate: ",oldLR)
            print("New Learning Rate: ",oldLR/mdlParams['LRstep'])
            update_op = lr.assign(oldLR/mdlParams['LRstep'])
            sess.run(update_op)
            if eval_set == 'valInd':
                mdlParams['lastLRUpdate'] = step
            else:
                mdlParams['lastLRUpdate'] = mdlParams['lastLRUpdate']+1
# After training: evaluate test set
# First, restore model with moving averages of vars
# Reset graph
tf.reset_default_graph()
# Rebuild placeholders
placeholders['X'] = tf.placeholder("float", input_size,name='X')
placeholders['Y'] = tf.placeholder("float", [None, len(mdlParams['tar_range'])],name='Y')
placeholders['train_state'] = tf.placeholder(tf.bool,name='train_state') 
# Rebuild graph
model_function = models.getModel(mdlParams,placeholders)
prediction = model_function(placeholders['X'])
# Rebuild loss op
loss_op = tf.reduce_mean(tf.square(prediction-placeholders['Y']))
# Get moving average varibales
variables_to_restore = variable_averages.variables_to_restore(slim.get_model_variables())
# Get the saver to restore them
saver = tf.train.Saver(max_to_keep=0,var_list=variables_to_restore)
# Session config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# Restore
with tf.Session(config=config) as sess:
    saver.restore(sess, mdlParams['saveDir'] + '/checkpoint-' + str(mdlParams['lastBestInd']))
    if 'testInd' in mdlParams:
        eval_set_final = 'testInd'
    else:
        eval_set_final = 'valInd'
    loss, mae_mean, mae_std, rmae_mean, rmae_std, acc = utils.getErrRegression(mdlParams, eval_set_final, sess, loss_op, prediction, getBatch, placeholders, feed_list_inference)
    print("Result on",eval_set_final," set with moving average variables")
    print("-------------------------------------------------")
    print("Loss: ",loss," MAE: ",mae_mean,"+-",mae_std)
    print("Relative MAE: ",rmae_mean,"+-",rmae_std," aCC: ",acc)