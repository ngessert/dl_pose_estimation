Implementation of the models used in the paper:

Nils Gessert, Matthias Schlüter, Alexander Schlaefer,  
A deep learning approach for pose estimation from volumetric OCT data,  
Medical Image Analysis,  
Volume 46,  
2018,  
Pages 162-179,  
ISSN 1361-8415,  
https://doi.org/10.1016/j.media.2018.03.002.  
(http://www.sciencedirect.com/science/article/pii/S1361841518300732)  
Keywords: 3D convolutional neural networks; 3D deep learning; Pose estimation; Optical coherence tomography  

Requirements:  
Working Tensorflow installation (tested with 1.5)

This environment is structured as follows:  
- train.py is the main script for training execution
- models.py contains model definitions, i.e. ResNetA3D, ResNetB3D, ResNeXt3D and Inception3D
- utils.py contains helper functions, e.g. for obtaining batches and evaluating errors
- eval.py script for quickly evaluating a model after training with this environment
- /cfgs/examples/ contains configurations for training

Before training, the respective cfg file needs to be configured. The pathes need to be changed
where the model is stored and where the training data is to be loaded from. Furthermore, the image data and labels need to be loaded
to python in the cfg file (see example). Lastly, training/validation/test set need to be defined as a sets of indices.

Note, that the implementation of ResNeXt3D is currently quite inefficient. If grouped convolutions are added to tensorflow, we will update the code.

Execute training:

python -u train.py examples.example_inception |& tee "log_output.txt"

Execute evaluation:

python -u eval.py examples.example_inception /path/to/saved/model

The second argument is the path where the trained model was saved by train.py.
