import tensorflow as tf
from tensorflow.python.ops import math_ops
import tensorflow.contrib.slim as slim
from tensorflow.contrib import rnn
import numpy as np
import functools
import h5py
import math

def resneta_block(x, numFmOut, stride):
    """Defines a single resnetA block, according to paper
    Args: 
      x: block input, 5D tensor
      base_fm: base number of feature maps in the block
    Returns:
      output: 5D tensor, output of the block 
    """
    # Number of input fms
    numFmIn = x.get_shape().as_list()[-1]
    # Determine if its a reduction
    if numFmOut > numFmIn:
        increase_dim = True
    else:
        increase_dim = False
    # First 3x3 layer
    with tf.variable_scope('conv3x3x3_1'):
        layer = slim.convolution(x,numFmOut,3,stride=stride)
    # Second 3x3 layer, no activation, only bnorm
    with tf.variable_scope('conv3x3x3_2'):
        layer = slim.convolution(layer,numFmOut,3,stride=1, activation_fn=None)
    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    # depth of input layers
    adjusted_input = x
    if stride == 2:
        # take care of 1D<->2D<->3D
        if len(x.get_shape().as_list()) == 5:
            adjusted_input = tf.nn.pool(adjusted_input,[2,2,2], "AVG", padding='SAME', strides=[2,2,2])
        elif len(x.get_shape().as_list()) == 4:
            adjusted_input = tf.nn.pool(adjusted_input,[2,2], "AVG", padding='SAME', strides=[2,2])
        else:
            adjusted_input = tf.nn.pool(adjusted_input,[2], "AVG", padding='SAME', strides=[2])
    if increase_dim:
        lower_pad = math.ceil((numFmOut-numFmIn)/2)
        upper_pad = (numFmOut-numFmIn)-lower_pad
        # take care of 1D<->2D<->3D
        if len(x.get_shape().as_list()) == 5:
            adjusted_input = tf.pad(adjusted_input, [[0, 0], [0, 0], [0, 0], [0,0], [lower_pad,upper_pad]])
        elif len(x.get_shape().as_list()) == 4:
            adjusted_input = tf.pad(adjusted_input, [[0, 0], [0, 0], [0, 0], [lower_pad,upper_pad]])
        else:
            adjusted_input = tf.pad(adjusted_input, [[0, 0], [0, 0], [lower_pad,upper_pad]])
    # Residual connection + activation
    output = tf.nn.relu(adjusted_input + layer)
    return output

def resnetb_block(x, numFmOut, bottleneck_size, stride):
    """Defines a single resnetB block, according to paper
    Args: 
      x: block input, 5D tensor
      base_fm: base number of feature maps in the block
    Returns:
      output: 5D tensor, output of the block 
    """
    # Number of input fms
    numFmIn = x.get_shape().as_list()[-1]
    # Determine if its a reduction
    if numFmOut > numFmIn:
        increase_dim = True
    else:
        increase_dim = False
    # First 1x1 layer
    with tf.variable_scope('conv1x1x1_1'):
        layer = slim.convolution(x,bottleneck_size,1,stride=1)
    # Second 3x3 layer, apply stride here
    with tf.variable_scope('conv3x3x3_2'):
        layer = slim.convolution(layer,bottleneck_size,3,stride=stride)
    # Third layer, restore FM size
    with tf.variable_scope('conv1x1x1_3'):
        layer = slim.convolution(layer,numFmOut,1,stride=1, activation_fn=None)
    # When the channels of input layer and conv2 does not match, add zero pads to increase the
    # depth of input layers
    adjusted_input = x
    if stride == 2:
        # take care of 1D<->2D<->3D
        if len(x.get_shape().as_list()) == 5:
            adjusted_input = tf.nn.pool(adjusted_input,[2,2,2], "AVG", padding='SAME', strides=[2,2,2])
        elif len(x.get_shape().as_list()) == 4:
            adjusted_input = tf.nn.pool(adjusted_input,[2,2], "AVG", padding='SAME', strides=[2,2])
        else:
            adjusted_input = tf.nn.pool(adjusted_input,[2], "AVG", padding='SAME', strides=[2])
    if increase_dim:
        lower_pad = math.ceil((numFmOut-numFmIn)/2)
        upper_pad = (numFmOut-numFmIn)-lower_pad
        # take care of 1D<->2D<->3D
        if len(x.get_shape().as_list()) == 5:
            adjusted_input = tf.pad(adjusted_input, [[0, 0], [0, 0], [0, 0], [0,0], [lower_pad,upper_pad]])
        elif len(x.get_shape().as_list()) == 4:
            adjusted_input = tf.pad(adjusted_input, [[0, 0], [0, 0], [0, 0], [lower_pad,upper_pad]])
        else:
            adjusted_input = tf.pad(adjusted_input, [[0, 0], [0, 0], [lower_pad,upper_pad]])
    # Residual connection + activation
    output = tf.nn.relu(adjusted_input + layer)
    return output

def resnext_block(x, numFmOut, bottleneck_size, stride, cardinality):
    """Defines a single resnext block, according to paper
    Args: 
      x: block input, 5D tensor
      numFmOut: int, number of feature maps to be outputted
      bottleneck_size: int, number of feature maps for every paths
      stride: int, stride for the 3x3x3 convolutions
      cardinality: int, number of paths
    Returns:
      output: 5D tensor, output of the block 
    """
    # Number of input fms
    numFmIn = x.get_shape().as_list()[-1]
    # Determine if its a reduction
    if numFmOut > numFmIn:
        increase_dim = True
    else:
        increase_dim = False
    # Split into paths
    all_paths = []
    for i in range(cardinality):
        # First, 1x1 to bring FMs down to bottleneck size
        with tf.variable_scope('conv1x1x1_%d'%(i)):
            layer = slim.convolution(x,bottleneck_size,1,stride=1)
        # Then, 3x3, apply stride
        with tf.variable_scope('conv3x3x3_%d'%(i)):
            layer = slim.convolution(layer,bottleneck_size,3,stride=stride)
        # Add to list of paths
        all_paths.append(layer)
    # Concat all paths
    layer = tf.concat(all_paths,axis=4,name='concat_paths')
    # Restore FM size from concatenated paths
    with tf.variable_scope('conv1x1x1_restore'):
        layer = slim.convolution(layer,numFmOut,1,stride=1, activation_fn=None)
    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    # depth of input layers
    adjusted_input = x
    if stride == 2:
        # take care of 1D<->2D<->3D
        if len(x.get_shape().as_list()) == 5:
            adjusted_input = tf.nn.pool(adjusted_input,[2,2,2], "AVG", padding='SAME', strides=[2,2,2])
        elif len(x.get_shape().as_list()) == 4:
            adjusted_input = tf.nn.pool(adjusted_input,[2,2], "AVG", padding='SAME', strides=[2,2])
        else:
            adjusted_input = tf.nn.pool(adjusted_input,[2], "AVG", padding='SAME', strides=[2])
    if increase_dim:
        lower_pad = math.ceil((numFmOut-numFmIn)/2)
        upper_pad = (numFmOut-numFmIn)-lower_pad
        # take care of 1D<->2D<->3D
        if len(x.get_shape().as_list()) == 5:
            adjusted_input = tf.pad(adjusted_input, [[0, 0], [0, 0], [0, 0], [0,0], [lower_pad,upper_pad]])
        elif len(x.get_shape().as_list()) == 4:
            adjusted_input = tf.pad(adjusted_input, [[0, 0], [0, 0], [0, 0], [lower_pad,upper_pad]])
        else:
            adjusted_input = tf.pad(adjusted_input, [[0, 0], [0, 0], [lower_pad,upper_pad]])
    # Residual connection + activation
    output = tf.nn.relu(adjusted_input + layer)
    return output


def inception_block(x, inception_dims, stride, scale, last=False):
    """Defines a single inception block, according to paper
    Args: 
      x: block input, 5D tensor
      inception_dims: 1D array, number of feature maps for unit in the block
      stride: int, contains the stride of the core convolutions, to be used for resizing the input
      scale: scale of the residual, see paper
      last: boolean, indicates whether this is the last block in a chain
    Returns:
      output: 5D tensor, output of the block 
    """
    # First: 1x1 layer
    with tf.variable_scope('conv1x1x1_1'):
        conv1x1x1_1 = slim.convolution(x,inception_dims[0],1,stride=stride)
    # Second: 1x1 with followed 3x3
    with tf.variable_scope('conv1x1x1_2'):
        conv1x1x1_2 = slim.convolution(x,inception_dims[1],1)
    with tf.variable_scope('conv3x3x3_2'):
        conv3x3x3_2 = slim.convolution(conv1x1x1_2,inception_dims[2],3,stride=stride)
    # Third: 1x1 with followed 3x3 3x3
    with tf.variable_scope('conv1x1x1_3'):
        conv1x1x1_3 = slim.convolution(x,inception_dims[3],1)
    with tf.variable_scope('conv3x3x3_3_1'):
        conv3x3x3_3_1 = slim.convolution(conv1x1x1_3,inception_dims[4],3)
    with tf.variable_scope('conv3x3x3_3_2'):
        conv3x3x3_3_2 = slim.convolution(conv3x3x3_3_1,inception_dims[4],3,stride=stride)
    # Concat
    output = tf.concat([conv1x1x1_1,conv3x3x3_2,conv3x3x3_3_2],4)
    # Resize input for residual connections
    if stride == 1:
        # Expand concat tensor to original size
        with tf.variable_scope('expand_output'):
            expanded_output = slim.convolution(output, x.get_shape().as_list()[-1], 1, activation_fn=None)
            # Residual connection with scale
            if last:
                output = scale*expanded_output + x
            else:
                output = scale*expanded_output + x
            output = tf.nn.relu(output)
    else: 
        # This is a reduction block, therefore adjust input instead
        pooled_input = slim.layers.avg_pool3d(x, 2)
        lower_pad = math.ceil((output.get_shape().as_list()[-1]-x.get_shape().as_list()[-1])/2)
        upper_pad = (output.get_shape().as_list()[-1]-x.get_shape().as_list()[-1])-lower_pad
        # Pad
        adjusted_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [0,0], [lower_pad,upper_pad]])
        # Residual connection with scale
        output = scale*output + adjusted_input
    return output

def ResNetA3D(x, mdlParams, placeholders=None):
    """Defines the ResNetA3D architecture from the paper "A Deep Learning Approach for Pose Estimation from Volumetric OCT Data"
    Args:
      x: 5D input tensor, usually a placeholder of shape [batchSize, width, height, depth, channel]
      mdlParams: dictionary, contains model configuration
      is_training: boolean, indicates if it is training or evaluation
    Returns:
      output: 2D tensor of shape [batchSize, numberOfOutputs]
    """
    with tf.variable_scope('ResNetA3D'):
        with slim.arg_scope([slim.convolution], padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01), normalizer_fn=slim.batch_norm, normalizer_params={'is_training':placeholders['train_state'], 'epsilon':0.0001, 'decay':0.9,  'center':True, 'scale':True, 'activation_fn':None, 'updates_collections':tf.GraphKeys.UPDATE_OPS, 'fused': False}):
            # Initial part
            with tf.variable_scope('Initial'):
                layer = slim.convolution(x, 48, 3, stride=1, scope='conv1')
                layer = slim.convolution(layer, 64, 3, stride=2, scope='conv2')
                layer = slim.convolution(layer, 64, 3, stride=1, scope='conv3')
                layer = slim.convolution(layer, 64, 3, stride=1, scope='conv4')
                layer = slim.convolution(layer, 64, 3, stride=1, scope='conv5')
            # Resnet modules
            with tf.variable_scope('Resnet_modules'):
                # Initial output feature map size
                output_fm = mdlParams['ResNetA3D_FM']
                # Iterate through all modules
                for i in range(len(mdlParams['ResNetA3D_Size'])):
                    with tf.variable_scope('Module_%d'%(i)):
                        # Iterate through all blocks inside the module
                        for j in range(mdlParams['ResNetA3D_Size'][i]):
                            with tf.variable_scope('Block_%d'%(j)):
                                # Set desired output feature map dimension of the block and the desired stride for the first block in the module
                                if j==0:
                                    output_fm = 2*output_fm
                                    block_stride = mdlParams['ResNetA3D_Stride'][i]
                                else:
                                    block_stride = 1
                                layer = resneta_block(layer, output_fm, block_stride)
            # GAP for 1D,2D,3D
            if len(layer.get_shape().as_list()) == 5:
                layer = math_ops.reduce_mean(layer, axis=[1,2,3], keep_dims = False, name='global_pool')
            elif len(layer.get_shape().as_list()) == 4:
                layer = math_ops.reduce_mean(layer, axis=[1,2], keep_dims = False, name='global_pool')
            else:
                layer = math_ops.reduce_mean(layer, axis=[1], keep_dims = False, name='global_pool')
            # Dense output layer
            output = slim.layers.fully_connected(layer, len(mdlParams['tar_range']), activation_fn=None)
    return output

def ResNetB3D(x, mdlParams, placeholders=None):
    """Defines the ResNetB3D architecture from the paper "A Deep Learning Approach for Pose Estimation from Volumetric OCT Data"
    Args:
      x: 5D input tensor, usually a placeholder of shape [batchSize, width, height, depth, channel]
      mdlParams: dictionary, contains model configuration
      is_training: boolean, indicates if it is training or evaluation
    Returns:
      output: 2D tensor of shape [batchSize, numberOfOutputs]
    """
    with tf.variable_scope('ResNetB3D'):
        with slim.arg_scope([slim.convolution], padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01), normalizer_fn=slim.batch_norm, normalizer_params={'is_training':placeholders['train_state'], 'epsilon':0.0001, 'decay':0.9,  'center':True, 'scale':True, 'activation_fn':None, 'updates_collections':tf.GraphKeys.UPDATE_OPS, 'fused': False}):
            # Initial part
            with tf.variable_scope('Initial'):
                layer = slim.convolution(x, 48, 3, stride=1, scope='conv1')
                layer = slim.convolution(layer, 64, 3, stride=2, scope='conv2')
                layer = slim.convolution(layer, 64, 3, stride=1, scope='conv3')
                layer = slim.convolution(layer, 64, 3, stride=1, scope='conv4')
                layer = slim.convolution(layer, 64, 3, stride=1, scope='conv5')
            # Resnet modules
            with tf.variable_scope('Resnet_modules'):
                # Initial output feature map size
                output_fm = mdlParams['ResNetB3D_FM']
                # Initial feature map sizes for bottleneck
                reduced_fm = mdlParams['ResNetB3D_Red_FM']
                # Iterate through all modules
                for i in range(len(mdlParams['ResNetB3D_Size'])):
                    with tf.variable_scope('Module_%d'%(i)):
                        # Iterate through all blocks inside the module
                        for j in range(mdlParams['ResNetB3D_Size'][i]):
                            with tf.variable_scope('Block_%d'%(j)):
                                # Set desired output feature map dimension of the block and the desired stride for the first block in the module
                                if j==0:
                                    output_fm = 2*output_fm
                                    reduced_fm = 2*reduced_fm
                                    block_stride = mdlParams['ResNetB3D_Stride'][i]
                                else:
                                    block_stride = 1
                                layer = resnetb_block(layer, output_fm, reduced_fm, block_stride)
            # GAP for 1D,2D,3D
            if len(layer.get_shape().as_list()) == 5:
                layer = math_ops.reduce_mean(layer, axis=[1,2,3], keep_dims = False, name='global_pool')
            elif len(layer.get_shape().as_list()) == 4:
                layer = math_ops.reduce_mean(layer, axis=[1,2], keep_dims = False, name='global_pool')
            else:
                layer = math_ops.reduce_mean(layer, axis=[1], keep_dims = False, name='global_pool')
            # Dense output layer
            output = slim.layers.fully_connected(layer, len(mdlParams['tar_range']), activation_fn=None)
    return output

def ResNeXt3D(x, mdlParams, placeholders=None):
    """Defines the ResNetB3D architecture from the paper "A Deep Learning Approach for Pose Estimation from Volumetric OCT Data"
    Args:
      x: 5D input tensor, usually a placeholder of shape [batchSize, width, height, depth, channel]
      mdlParams: dictionary, contains model configuration
      is_training: boolean, indicates if it is training or evaluation
    Returns:
      output: 2D tensor of shape [batchSize, numberOfOutputs]
    """
    with tf.variable_scope('ResNetA3D'):
        with slim.arg_scope([slim.convolution], padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01), normalizer_fn=slim.batch_norm, normalizer_params={'is_training':placeholders['train_state'], 'epsilon':0.0001, 'decay':0.9,  'center':True, 'scale':True, 'activation_fn':None, 'updates_collections':tf.GraphKeys.UPDATE_OPS, 'fused': False}):
            # Initial part
            with tf.variable_scope('Initial'):
                layer = slim.convolution(x, 48, 3, stride=1, scope='conv1')
                layer = slim.convolution(layer, 64, 3, stride=2, scope='conv2')
                layer = slim.convolution(layer, 64, 3, stride=1, scope='conv3')
                layer = slim.convolution(layer, 64, 3, stride=1, scope='conv4')
                layer = slim.convolution(layer, 64, 3, stride=1, scope='conv5')
            # Resnet modules
            with tf.variable_scope('Resnet_modules'):
                # Initial output feature map size
                output_fm = mdlParams['ResNeXt3D_FM']
                # Initial feature map sizes for bottleneck
                reduced_fm = mdlParams['ResNeXt3D_Red_FM']
                # Iterate through all modules
                for i in range(len(mdlParams['ResNeXt3D_Size'])):
                    with tf.variable_scope('Module_%d'%(i)):
                        # Iterate through all blocks inside the module
                        for j in range(mdlParams['ResNeXt3D_Size'][i]):
                            with tf.variable_scope('Block_%d'%(j)):
                                # Set desired output feature map dimension of the block and the desired stride for the first block in the module
                                if j==0:
                                    output_fm = 2*output_fm
                                    reduced_fm = 2*reduced_fm
                                    block_stride = mdlParams['ResNeXt3D_Stride'][i]
                                else:
                                    block_stride = 1
                                layer = resnext_block(layer, output_fm, reduced_fm, block_stride, mdlParams['cardinality'])
            # GAP for 1D,2D,3D
            if len(layer.get_shape().as_list()) == 5:
                layer = math_ops.reduce_mean(layer, axis=[1,2,3], keep_dims = False, name='global_pool')
            elif len(layer.get_shape().as_list()) == 4:
                layer = math_ops.reduce_mean(layer, axis=[1,2], keep_dims = False, name='global_pool')
            else:
                layer = math_ops.reduce_mean(layer, axis=[1], keep_dims = False, name='global_pool')
            # Dense output layer
            output = slim.layers.fully_connected(layer, len(mdlParams['tar_range']), activation_fn=None)
    return output

def Inception3D(x, mdlParams, placeholders=None):
    """Defines the Inception3D architecture from the paper "A Deep Learning Approach for Pose Estimation from Volumetric OCT Data"
    Args:
      x: 5D input tensor, usually a placeholder of shape [batchSize, width, height, depth, channel]
      mdlParams: dictionary, contains model configuration
      is_training: boolean, indicates if it is training or evaluation
    Returns:
      output: 2D tensor of shape [batchSize, numberOfOutputs]
    """
    with tf.variable_scope('Inception3D'):
        with slim.arg_scope([slim.convolution], padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01), normalizer_fn=slim.batch_norm, normalizer_params={'is_training':placeholders['train_state'], 'epsilon':0.0001, 'decay':0.9,  'center':True, 'scale':True, 'activation_fn':None, 'updates_collections':tf.GraphKeys.UPDATE_OPS, 'fused': False}):
            # Initial part
            with tf.variable_scope('Initial'):
                layer = slim.convolution(x, 48, 3, stride=1, scope='conv1')
                layer = slim.convolution(layer, 64, 3, stride=2, scope='conv2')
                layer = slim.convolution(layer, 64, 3, stride=1, scope='conv3')
                layer = slim.convolution(layer, 64, 3, stride=1, scope='conv4')
                layer = slim.convolution(layer, 64, 3, stride=1, scope='conv5')
            # Inception modules
            with tf.variable_scope('Inception_Modules'):
                # Iterate through all modlues
                for i in range(len(mdlParams['num_inception_blocks'])):
                    with tf.variable_scope('Module_%d'%(i)):
                        # Save for long-range connections
                        module_input = layer
                        # Input feature map size for the first block, needed for long range connections
                        input_size = module_input.get_shape().as_list()[-1]
                        # First, apply reduction block
                        with tf.variable_scope('Reduction_Block'):
                            layer = inception_block(layer, mdlParams['inception_dims_reduction'][i,:], stride=2, scale=mdlParams['inception_block_scale'])
                        # Input size for the rest of the modules, needed for long range connections
                        red_fm_size = mdlParams['inception_dims_reduction'][i,0]+mdlParams['inception_dims_reduction'][i,2]+mdlParams['inception_dims_reduction'][i,4]
                        # Then, add normal inception blocks
                        for j in range(mdlParams['num_inception_blocks'][i]):
                            with tf.variable_scope('Normal_Block_%d'%(j)):
                                layer = inception_block(layer, mdlParams['inception_dims'][i,:], stride=1, scale=mdlParams['inception_block_scale'], last=(j==mdlParams['num_inception_blocks'][i]-1))
                        # If desired, add long range connection from the input
                        if mdlParams['long_range_connection'][i] > 0:
                            # Resize input, depending on connection type
                            # If long-range residual connections are used
                            if mdlParams['long_range_connection'][i] == 1:
                                with tf.variable_scope('resize_module'):
                                    adjusted_input = slim.convolution(module_input, red_fm_size, 1, stride=2)
                                # Add scaled residual connection
                                layer = mdlParams['module_scale']*layer + adjusted_input
                            # If long-range dense connections are used
                            elif mdlParams['long_range_connection'][i] == 2:
                                pooled_input = slim.layers.avg_pool3d(module_input, 2)
                                lower_pad = math.ceil((red_fm_size-input_dim)/2)
                                upper_pad = (red_fm_size-input_dim)-lower_pad
                                # Pad
                                adjusted_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [0,0], [lower_pad,upper_pad]])
                                # Concat and adjust size with conv
                                target_size = layer.get_shape().as_list()[-1]
                                layer = tf.concat([layer,adjusted_input],4)
                                layer = slim.convolution(layer, target_size, 1, scope='long_range_resize')
            # GAP for 1D,2D,3D
            if len(layer.get_shape().as_list()) == 5:
                layer = math_ops.reduce_mean(layer, axis=[1,2,3], keep_dims = False, name='global_pool')
            elif len(layer.get_shape().as_list()) == 4:
                layer = math_ops.reduce_mean(layer, axis=[1,2], keep_dims = False, name='global_pool')
            else:
                layer = math_ops.reduce_mean(layer, axis=[1], keep_dims = False, name='global_pool')
            # Dense output layer
            output = slim.layers.fully_connected(layer, len(mdlParams['tar_range']), activation_fn=None)
    return output 

model_map = {'ResNetA3D': ResNetA3D,
             'ResNetB3D': ResNetB3D,
             'ResNeXt3D': ResNeXt3D,
             'Inception3D': Inception3D,
               }

def getModel(mdlParams, placeholders):
  """Returns a function for a model
  Args:
    mdlParams: dictionary, contains configuration
    is_training: bool, indicates whether training is active
  Returns:
    model: A function that builds the desired model
  Raises:
    ValueError: If model name is not recognized.
  """
  if mdlParams['model_type'] not in model_map:
    raise ValueError('Name of model unknown %s' % mdlParams['model_type'])
  func = model_map[mdlParams['model_type']]
  @functools.wraps(func)
  def model(x):
      return func(x, mdlParams, placeholders)
  return model