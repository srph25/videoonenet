import numpy as np
from utils.pil import fromimage, toimage, imresize, imread, imsave
from keras.datasets import cifar10
from utils.utils import LinearInverseVideoSequence
from datasets.mnistrotated import MNISTRotatedDataset


class CIFAR10ScannedDataset(MNISTRotatedDataset):

    def __init__(self, config, seed):
        """Constructor.
        """
        
        self.config = config
        
        (X_train, _), (X_test, _) = cifar10.load_data()
        X_train = X_train.reshape(-1, 32, 32, 3)
        X_test = X_test.reshape(-1, 32, 32, 3)
        X_train = X_train.astype(self.config['dtype'])
        X_test = X_test.astype(self.config['dtype'])
        N = ((32 - self.config['size']) // self.config['stride']) + 1
        X_train_scan = np.zeros((X_train.shape[0], N ** 2, self.config['size'], self.config['size'], 3))
        X_test_scan = np.zeros((X_test.shape[0], N ** 2, self.config['size'], self.config['size'], 3))
        for i in range(len(X_train)):
            for t1 in range(N):
                for t2 in (range(N) if (np.mod(t1, 2) == 0) else range(N - 1, -1, -1)):
                    X_train_scan[i, t1 * N + (t2 if (np.mod(t1, 2) == 0) else (N - 1 - t2)), :, :, :] = X_train[i, (t2 * self.config['stride']):(t2 * self.config['stride'] + self.config['size']), (t1 * self.config['stride']):(t1 * self.config['stride'] + self.config['size']), :]
        for i in range(len(X_test)):
            for t1 in range(N):
                for t2 in (range(N) if (np.mod(t1, 2) == 0) else range(N - 1, -1, -1)):
                    X_test_scan[i, t1 * N + (t2 if (np.mod(t1, 2) == 0) else (N - 1 - t2)), :, :, :] = X_test[i, (t2 * self.config['stride']):(t2 * self.config['stride'] + self.config['size']), (t1 * self.config['stride']):(t1 * self.config['stride'] + self.config['size']), :]
        X_train_scan /= 255.
        X_test_scan /= 255.

        num_train = round(0.9 * len(X_train_scan))
        perm = np.random.permutation(len(X_train_scan))
        X_val_scan = X_train_scan[perm[num_train:]]
        X_train_scan = X_train_scan[perm[:num_train]]
        
        self.X_train = X_train_scan
        self.X_val = X_val_scan
        self.X_test = X_test_scan
        seq_config = self.config
        self.generate_sequences(self.config.copy(), seed=seed)
        print(self.X_train.shape, self.X_val.shape, self.X_test.shape)

