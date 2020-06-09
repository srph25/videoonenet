VideoOneNet: Bidirectional Convolutional Recurrent OneNet with Trainable Data Steps for Video Processing
Python source code.
Code is mostly self-explanatory via file, variable and function names; but more complex lines are commented.
Designed to require minimal setup overhead, using as much Keras and sacred integration and reusability as possible.


Dependencies:
python 3.7.6
cudatoolkit 10.0.130
cudnn 7.6.4
h5py 2.10.0
ipython 7.9.0
keras 2.3.0
matplotlib 3.1.1
numpy 1.18.1
pillow 5.1.0
pywavelets 1.1.1
sacred 0.8.1
scikit-learn 0.21.3
scipy 1.3.2
tf-nightly-gpu 1.14.1.dev20190606
tqdm 4.38.0


Directory and file structure:
algorithms/
           kerasvideoonenet.py : base class, our videoonenet method with 2 contributions
           kerasvideoonenet_admm.py : subclass, the original onenet baseline method
           numpyvideowaveletsparsity_admm.py : subclass, the wavelet sparsity baseline method (CPU-only)
datasets/
         mnistrotated.py : base class, loads Rotated MNIST data set and generates linear inverse problems on the fly
         cifar10scanned.py : subclass, same but for Scanned CIFAR-10
         ucf101.py : subclass, same but for UCF-101
experiments/
            mnistrotated.py : config file for hyperparameters, loads Rotated MNIST data set and an algorithm, conducts experiment; requires ~8Gb GPUs for training
            cifar10scanned : same, but for Scanned CIFAR-10; requires ~12Gb GPUs for training
            ucf101 : same, but for UCF-101, requires 2*24Gb GPUs for training
results/ : experimental results will be saved to this directory with sacred package
utils/
      layers.py : custom Keras layer classes, including
               /ConvMinimalRNN2D : the convolutional minimal recurrent layer
               /InstanceNormalization : the instance normalization layer
               /UnrolledOptimization : the layer responsible for end-to-end trainable ADMM iterations, the core of our algorithms
      ops.py : custom Keras/Tensorflow operations, including
            /elu_like : the activation function
            /batch_pinv : batched Moore-Penrose pseudoinverse computation
      pil.py : functions for backwards compatibility for saving all kinds of figures
      plot.py : functions for saving video frame figures
      problems.py : functions for the linear inverse problem set, each generating the matrix A^{(n)} and an auxiliary shape for saving figures
      utils.py : additional things
              /LinearInverseVideoSequence : Keras Sequence subclass generating random videos and linear inverse problems from the given problem set


Usage:
Example : execute in root directory : ipython3 experiments/mnistrotated.py with videoonenet seed=123
In general : ipython3 experiments/database.py with algorithm optional_config seed=number
          where algorithm is either:
                                 videoonenet : our videoonenet method with 2 contributions
                                 videoonenetadmm : the original onenet baseline method
                                 videowaveletsparsityadmm : the wavelet sparsity baseline method
          and optional_config is either nothing (convolutional minimal recurrent layer enabled by default), or:
                                                                  rnn : convolutional minimal recurrent layer enabled
                                                                  nornn : convolutional minimal recurrent layer disabled
          seed : 123 in all of our experiments, should yield very similar numbers as in the table of our paper


Note:
Using tensorflow-gpu 1.13.0 produces severe memory leaks with Keras Sequence classes. Please consider using the version mentioned above.

