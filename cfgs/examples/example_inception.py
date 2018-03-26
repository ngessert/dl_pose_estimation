import os
import sys
import numpy as np
import h5py

mdlParams = {}
# Defines which part of the pose is trained on. See data section below
mdlParams['tar_range'] = np.array([0,1,2])
# Input size, without batchsize/channel
mdlParams['input_size'] = np.array([64,64,16])
# Save summaries and model here
#mdlParams['saveDir'] = '/path/to/savedir/inception'
mdlParams['saveDir'] = '/home/Gessert/data/newMIA/inception'
# Data is loaded from here
#mdlParams['dataDir'] = '/path/to/data/'
mdlParams['dataDir'] = '/home/Gessert/data/newMIA/'

### Model Selection ###
mdlParams['model_type'] = 'Inception3D'
mdlParams['batch_func_type'] = 'getBatch3D'

### Inception3D Parameters ###
# Number of normal inception blocks per module
mdlParams['num_inception_blocks'] = np.array([3,4])
# Parameters for the initial inception reduction block
mdlParams['inception_dims_reduction'] = np.array([[30,64,64,64,64],[40,86,86,86,86]])
# Parameters for the normal inception blocks
mdlParams['inception_dims'] = np.array([[20,42,42,42,42],[30,64,64,64,64]])
# Scale parameter for residual connection inside inception block
mdlParams['inception_block_scale'] = 0.8
# Scale parameter for long-range residual connection
mdlParams['module_scale'] = 0.5
# Long-range connection type, 1 -> residual 2-> dense 0 -> none
mdlParams['long_range_connection'] = np.array([1,1])

### Training Parameters ###
# Batch size
mdlParams['batchSize'] = 15
# Initial learning rate
mdlParams['learning_rate'] = 0.0005
# Lower learning rate after no improvement over 100 epochs
mdlParams['lowerLRAfter'] = 150
# If there is no validation set, start lowering the LR after X steps
mdlParams['lowerLRat'] = 350
# Divide learning rate by this value
mdlParams['LRstep'] = 2
# Maximum number of training iterations
mdlParams['training_steps'] = 10
# Display error every X steps
mdlParams['display_step'] = 5
# Scale?
mdlParams['scale_targets'] = True
# Peak at test error during training? (generally, dont do this!)
mdlParams['peak_at_testerr'] = True
# Decay of moving averages
mdlParams['moving_avg_var_decay'] = 0.99

### Data ###
# Example for loading the data into python. The format should be [B, W, H, D, C]
with h5py.File(mdlParams['dataDir'] + 'data.h5','r') as f:
    mdlParams['data'] = np.array(f['volumes'].value)
    mdlParams['targets'] = np.array(f['targets'].value)

# Load indices or define them here
with h5py.File(mdlParams['dataDir'] + 'indices.h5','r') as f:
    mdlParams['trainInd'] = np.array(f['trainInd'].value,dtype=int)
    mdlParams['valInd'] = np.array(f['testInd'].value,dtype=int)
    mdlParams['testInd'] = np.array(f['testInd'].value,dtype=int)

