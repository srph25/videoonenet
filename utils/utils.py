import numpy as np
import scipy
from utils.pil import fromimage, toimage, imresize, imread, imsave
#from scipy.misc import imresize, imread
import os
import h5py
import tqdm
from keras.utils import Sequence, to_categorical
from utils.problems import *
import time


class LinearInverseVideoSequence(Sequence):
    def __init__(self, config, train_val_test, videos, grayscale=False, seed=None):
        self.config = config
        self.train_val_test = train_val_test
        self.videos = videos
        if isinstance(self.videos, list):
            self.get_metas()
        self.grayscale = grayscale
        self.precomputing = False
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            
    def __len__(self):
        # Get steps per epoch.
       return int(np.ceil(len(self.videos) / self.config['batch_size']))

    def __getitem__(self, idx):
        batch_start = idx * self.config['batch_size']
        batch_end = np.min([batch_start + self.config['batch_size'], len(self.videos)])
        if isinstance(self.videos, list):
            metas = self.metas[batch_start:batch_end]
            lens = [v[0] for v in metas]
        X = []
        for v, video in enumerate(self.videos[batch_start:batch_end]):
            if isinstance(self.videos, list) and isinstance(video, str):
                frame_start = self.rng.randint(lens[v] - self.config['frames']) if (lens[v] > self.config['frames']) else 0
                frame_end = np.min([frame_start + self.config['frames'], lens[v]])
                frames = self.build_frames(video, metas[v], frame_start=frame_start, frame_end=frame_end)
            elif isinstance(video, np.ndarray):
                frame_start = self.rng.randint(video.shape[0] - self.config['frames']) if (video.shape[0] > self.config['frames']) else 0
                frame_end = np.min([frame_start + self.config['frames'], video.shape[0]])
                frames = video[frame_start:frame_end]
            X.append(frames)
        X = np.array(X, dtype=self.config['dtype'])
        y_batch = []
        A_batch = []
        for v, video in enumerate(self.videos[batch_start:batch_end]):
            problem = self.config['problems'][self.rng.randint(len(self.config['problems']))]
            shp = (1,) + X[v].shape
            try:
                A, shp_out = eval(problem[0])(shp, rng=self.rng, **(problem[1]))
            except:
                A, shp_out = eval(problem[0])(shp, **(problem[1]))
            A = A.astype(self.config['dtype'])
            y = A.dot(np.reshape(X[v], (-1, 1))) + self.rng.randn(A.shape[0], 1) * (0.1 if problem[0] == 'inpaint' else 0.0) # add noise
            y = y.flatten()
            y_batch.append(y.copy())
            A_batch.append(A.copy())
            del y, A
        max_len = np.max([A.shape[0] for A in A_batch])
        for i in range(len(y_batch)):
            y_batch[i] = np.concatenate([y_batch[i].copy(), np.zeros((max_len - y_batch[i].shape[0],) + y_batch[i].shape[1:], dtype=self.config['dtype'])], axis=0).astype(self.config['dtype'])
            A_batch[i] = np.concatenate([A_batch[i].copy(), np.zeros((max_len - A_batch[i].shape[0],) + A_batch[i].shape[1:], dtype=self.config['dtype'])], axis=0).astype(self.config['dtype'])
        y_batch = np.array(y_batch, dtype=self.config['dtype'])
        A_batch = np.array(A_batch, dtype=self.config['dtype'])
        return ([y_batch, A_batch], X)

    def get_meta(self, video):
        frames = self.get_frames(video)
        try: 
            img = imread(self.config['path'] + '/' + video.split('/')[0] + '/' + frames[0])
        except:
            img = imread(self.config['path'] + '/' + video + '/' + frames[0])
        frame_count, height, width = len(frames), img.shape[0], img.shape[1]
        return frame_count, height, width

    def get_metas(self):
        self.metas = []
        filename = self.config['path_split'] + '/' + self.train_val_test + 'list0' + str(self.config['split']) + '_meta.txt'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = f.readlines()
            self.videos = [d.split(' ')[0] for d in data]
            self.metas = [[int(d.split(' ')[1]), int(d.split(' ')[2]), int(d.split(' ')[3].split('\n')[0])] for d in data]
        else:
            with open(filename, 'w') as f:
                for video in tqdm.tqdm(self.videos):
                    meta = self.get_meta(video)
                    self.metas.append(list(meta))
                    print(video, meta)
                    f.write(video + ' ' + str(meta[0]) + ' ' + str(meta[1]) + ' ' + str(meta[2]) + '\n') 

    def get_frames(self, video):
        try:
            frames = [f for f in os.listdir(self.config['path'] + '/' + video.split('/')[0]) if ('jpg' in f and video.split('/')[1][:-4] in f)]
        except:
            frames = [f for f in os.listdir(self.config['path'] + '/' + video) if ('jpg' in f)]
        frames = np.sort(frames).tolist()
        return frames        

    def build_frames(self, video, meta, frame_start=None, frame_end=None):
        """Given a video name, build our sequence."""
        frame_count, height, width = meta
        if frame_start == None:
            frame_start = 0
        elif frame_start >= frame_count:
            return np.array([], dtype=self.config['dtype'])
        if frame_end == None:
            frame_end = frames
        elif frame_end >= frame_count:
            frame_end = frame_count

        if 'size_crop' in self.config.keys():
            row_start = self.rng.randint(height - self.config['size_crop'])
            col_start = self.rng.randint(width - self.config['size_crop'])
            row_end = row_start + self.config['size_crop']
            col_end = col_start + self.config['size_crop']
        else:
            row_start, col_start, row_end, col_end = 0, 0, height, width
        frames = self.get_frames(video)
        imgs = []
        for j in range(frame_start, frame_end):
            try: 
                img = imread(self.config['path'] + '/' + video.split('/')[0] + '/' + frames[j])
            except:
                img = imread(self.config['path'] + '/' + video + '/' + frames[0])
            img = img[row_start:row_end, col_start:col_end]
            img = imresize(img, (self.config['size'], self.config['size']))
            if self.grayscale is True:
                img = np.dot(img[...,:3], [0.299, 0.587, 0.114])[:, :, None]
            imgs.append(img)
        imgs = np.array(imgs, dtype=self.config['dtype']) / 255.
        return imgs

