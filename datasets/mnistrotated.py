import numpy as np
from utils.pil import fromimage, toimage, imresize, imread, imsave
from PIL import Image
from keras.datasets import mnist
from utils.utils import LinearInverseVideoSequence


class MNISTRotatedDataset():

    def __init__(self, config, seed):
        """Constructor.
        """
        
        self.config = config
        
        (X_train, _), (X_test, _) = mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        X_train = X_train.astype(self.config['dtype'])
        X_test = X_test.astype(self.config['dtype'])
        X_train_rot = np.zeros((X_train.shape[0], self.config['frames'], 
                                self.config['size'], self.config['size'], 1))
        X_test_rot = np.zeros((X_test.shape[0], self.config['frames'], self.config['size'], self.config['size'], 1))
        for i in range(len(X_train)):
            img = Image.fromarray(np.reshape(X_train[i, :, :, :].astype('float32'), (28, 28)))
            img = img.convert('L')
            phase = 360. * np.random.rand()
            for t in range(self.config['frames']):
                _img = img.rotate(phase + t * 360. / self.config['frames'], Image.BILINEAR).resize((28, 28), Image.BILINEAR).getdata()
                _img = np.reshape(np.array(_img, dtype=np.uint8), (28, 28, 1))[:self.config['size'], :self.config['size'], :]
                X_train_rot[i, t, :, :, :] = _img.astype(dtype=self.config['dtype'])
                
        for i in range(len(X_test)):
            img = Image.fromarray(np.reshape(X_test[i, :, :, :].astype('float32'), (28, 28)))
            img = img.convert('L')
            phase = 360. * np.random.rand()
            for t in range(self.config['frames']):
                _img = img.rotate(phase + t * 360. / self.config['frames'], Image.BILINEAR).resize((28, 28), Image.BILINEAR).getdata()
                _img = np.reshape(np.array(_img, dtype=np.uint8),  (28, 28, 1))[:self.config['size'], :self.config['size'], :]
                X_test_rot[i, t, :, :, :] = _img.astype(dtype=self.config['dtype'])
        X_train_rot /= 255.
        X_test_rot /= 255.

        num_train = round(0.9 * len(X_train_rot))
        perm = np.random.permutation(len(X_train_rot))
        X_val_rot = X_train_rot[perm[num_train:]]
        X_train_rot = X_train_rot[perm[:num_train]]
        
        self.X_train = X_train_rot
        self.X_val = X_val_rot
        self.X_test = X_test_rot
        self.generate_sequences(self.config.copy(), seed=seed)
        print(self.X_train.shape, self.X_val.shape, self.X_test.shape)
    
    def generate_sequences(self, config, seed):
        self.seq_train = LinearInverseVideoSequence(config, 'train', self.X_train, seed=seed)
        self.seq_val = LinearInverseVideoSequence(config, 'val', self.X_val, seed=(seed + 1)) # to ensure different noise matrix A
        self.seq_test = LinearInverseVideoSequence(config, 'test', self.X_test, seed=(seed + 2)) # to ensure different noise matrix A

