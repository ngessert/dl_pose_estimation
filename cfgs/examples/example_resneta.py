import os
import sys
import h5py
import numpy as np

mdlParams = {}
# Defines which part of the pose is trained on. See data section below
mdlParams['tar_range'] = np.array([0,1,2])
# Input size, without batchsize/channel
mdlParams['input_size'] = np.array([64,64,16])
# Save summaries and model here
mdlParams['saveDir'] = '/path/to/savedir/resnetA'
# Data is loaded from here
mdlParams['dataDir'] = '/path/to/data/'

### Model Selection ###
mdlParams['model_type'] = 'ResNetA3D'
mdlParams['batch_func_type'] = 'getBatch3D'

### ResNetA3D Parameters ###
# Defines number of modules and blocks per module
mdlParams['ResNetA3D_Size'] = [2,2]
# Defines the stride that is used at the beginning of each module
mdlParams['ResNetA3D_Stride'] = [2,2]
# Defines the base number of feature maps
mdlParams['ResNetA3D_FM'] = 64

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
mdlParams['scale_targets'] = False
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

