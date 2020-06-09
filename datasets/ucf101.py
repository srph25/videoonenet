import numpy as np
from keras.utils import to_categorical
from datasets.mnistrotated import MNISTRotatedDataset
from utils.utils import LinearInverseVideoSequence


class UCF101Dataset(MNISTRotatedDataset):

    def __init__(self, config, seed):
        """Constructor.
        """
        
        self.config = config
        
        with open(self.config['path_split'] + '/trainlist0' + str(self.config['split']) + '.txt', 'r') as f:
            videos_train = f.readlines()
        with open(self.config['path_split'] + '/testlist0' + str(self.config['split']) + '.txt', 'r') as f:
            videos_test = f.readlines()
        with open(self.config['path_split'] + '/classInd.txt', 'r') as f:
            classes = f.readlines()
        self.videos_train = [l.split(' ')[0] for l in videos_train]
        self.videos_test = [l.split('\n')[0].strip() for l in videos_test]
        self.classes = [l.split(' ')[1].split('\n')[0] for l in classes]
        self.generate_sequences(self.config.copy(), seed=seed)

    def generate_sequences(self, config, seed):
        videos_train_groups = [l.split('/')[1].split('.')[0].split('_')[2][1:3] for l in self.videos_train]
        videos_groups = np.unique(videos_train_groups)
        idxs_train = np.where([(True if videos_train_groups[j] in videos_groups[:-3] else False) for j in range(len(videos_train_groups))])[0]
        idxs_val = np.where([(True if videos_train_groups[j] in videos_groups[-3:] else False) for j in range(len(videos_train_groups))])[0]
        
        self.seq_train = LinearInverseVideoSequence(config, 'train', [self.videos_train[j] for j in idxs_train], seed=seed)
        self.seq_val = LinearInverseVideoSequence(config, 'val', [self.videos_train[j] for j in idxs_val], seed=(seed + 1)) # to ensure different noise matrix A
        self.seq_test = LinearInverseVideoSequence(config, 'test', self.videos_test, seed=(seed + 2)) # to ensure different noise matrix A

