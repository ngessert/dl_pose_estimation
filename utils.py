import functools
import numpy as np
import math
import sklearn.preprocessing

def getBatch3D(mdlParams, indices, batchID):
    """Helper function to get a batch from a set of indices.
    Args:
      mdlParams: dictionary, configuration file
      indices: string, either "trainInd", "valInd" or "testInd"
      batchID: int, batch number in the set of indices
    Returns:
      batch_x: A 5-D numpy array containing a batch of volumes
      batch_y: A 2-D numpy array containing the regression labels
    """
    # Determine if we are at the last batch, if yes, the batch is smaller
    if int(math.ceil(len(mdlParams[indices]) / mdlParams['batchSize'])) - 1 == batchID :
        bSize = len(mdlParams[indices]) - mdlParams['batchSize'] * batchID
    else:
        bSize = mdlParams['batchSize']
    # Define batch arrays
    batch_x = np.zeros([bSize, mdlParams['input_size'][0], mdlParams['input_size'][1], mdlParams['input_size'][2], 1])
    batch_y = np.zeros([bSize,len(mdlParams['tar_range'])])
    # Fill batch arrays
    batch_x[:,:,:,:,0] = mdlParams['data'][mdlParams[indices][mdlParams['batchSize']*batchID:(mdlParams['batchSize']*batchID+bSize)],:,:,:]-mdlParams['setMean']
    batch_y[:,:] = mdlParams['targets'][mdlParams[indices][mdlParams['batchSize']*batchID:(mdlParams['batchSize']*batchID+bSize)],:][:,mdlParams['tar_range'].astype(int)]
    # Scale targets
    if mdlParams['scale_targets']:
        batch_y = mdlParams['scaler'].transform(batch_y)
    return batch_x, batch_y


getBatch_map = {'getBatch3D': getBatch3D,
               }

def getBatchFunction(mdlParams):
  """Returns a function for a getBatch function
  Args:
    mdlParams: dictionary, contains configuration
  Returns:
    getBatch: A function that returns a batch for the network
  Raises:
    ValueError: If network name is not recognized.
  """
  if mdlParams['batch_func_type'] not in getBatch_map:
    raise ValueError('Name of getBatch function unknown %s' % mdlParams['batch_func_type'])
  func = getBatch_map[mdlParams['batch_func_type']]
  @functools.wraps(func)
  def getBatch(mdlParams, indices, batchID):
      return func(mdlParams, indices, batchID)
  return getBatch

def getErrRegression(mdlParams, indices, sess, loss_op, prediction, getBatch, placeholders, feed_list_inference):
    """Helper function to return the error of a set
    Args:
      mdlParams: dictionary, configuration file
      indices: string, either "trainInd", "valInd" or "testInd"
    Returns:
      loss: float, avg loss
      mae: float, mean average error
      mae_std: float, standard deviation of the per sample mae
      rmae: float, relative mean average error (divivded by targets' standard deviation)
      rmae_std: float, standard deviation of the per sample rmae
      acc: float, average correlation coefficient
    """
    # Set up sizes
    loss = np.zeros([len(mdlParams[indices])])
    predictions = np.zeros([len(mdlParams[indices]),len(mdlParams['tar_range'])])
    targets = np.zeros([len(mdlParams[indices]),len(mdlParams['tar_range'])])
    numBatches = int(math.ceil(len(mdlParams[indices])/mdlParams['batchSize']))
    for k in range(numBatches):
        feed_list_inference['X'], feed_list_inference['Y'] = getBatch(mdlParams,indices,k)
        # Take care of last batch being smaller
        if int(math.ceil(len(mdlParams[indices]) / mdlParams['batchSize'])) - 1 == k :
            bSize = len(mdlParams[indices]) - mdlParams['batchSize'] * k
        else:
            bSize = mdlParams['batchSize']
        targets[mdlParams['batchSize']*k:(mdlParams['batchSize']*k+bSize),:] = feed_list_inference['Y'] 
        loss[mdlParams['batchSize']*k:(mdlParams['batchSize']*k+bSize)], predictions[mdlParams['batchSize']*k:(mdlParams['batchSize']*k+bSize)] = sess.run([loss_op, prediction], feed_dict={placeholders[p]: feed_list_inference[p] for p in placeholders})
    # Transform targets and predictions
    if mdlParams['scale_targets']:
        targets = mdlParams['scaler'].inverse_transform(targets)
        predictions = mdlParams['scaler'].inverse_transform(predictions)
    # Error metrics
    # MAE
    mae = np.mean(np.abs(predictions-targets),1)
    mae_mean = np.mean(mae)
    mae_std = np.std(mae)
    # Relative MAE
    tar_std = np.std(targets,0)
    rmae_mean = np.mean(np.mean(np.abs(predictions-targets),0)/tar_std)
    rmae_std = np.mean(np.std(np.abs(predictions-targets),0)/tar_std)
    # Avg. Corr. Coeff.
    corr = np.corrcoef(np.transpose(predictions),np.transpose(targets))
    # Extract relevant components for aCC
    num_tar = len(mdlParams['tar_range'])
    acc = 0
    for k in range(num_tar):
        acc += corr[num_tar+k,k]
    acc /= num_tar
    return np.mean(loss), mae_mean, mae_std, rmae_mean, rmae_std, acc
