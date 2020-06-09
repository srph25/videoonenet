import numpy as np
import os
import datetime
from sacred import Experiment
from sacred.observers import FileStorageObserver
from datasets.mnistrotated import MNISTRotatedDataset
from algorithms.kerasvideoonenet import KerasVideoOneNet
from algorithms.kerasvideoonenet_admm import KerasVideoOneNetADMM
from algorithms.numpyvideowaveletsparsity_admm import NumpyVideoWaveletSparsityADMM

name = os.path.basename(__file__).split('.')[0]
ex = Experiment(name)
dt = datetime.datetime.now()
results_dir = 'results/' + name + '/' + '{y:04d}{mo:02d}{d:02d}{h:02d}{mi:02d}{s:02d}_{p:05d}'.format(y=dt.year, mo=dt.month, d=dt.day, h=dt.hour, mi=dt.minute, s=dt.second, p=os.getpid()) + '_' + os.uname()[1]
ex.observers.append(FileStorageObserver.create(results_dir))


@ex.config
def cfg():
    _data = {'batch_size': 8,
             'frames': 9,
             'size': 14,
             'dtype': 'float32',
             'problems_test': [('inpaint', {'drop_prob': 0.5}),
                               ('center', {'box_size_ratio': 0.4}),
                               ('block', {'box_size_ratio': 0.2,
                                          'total_box': 10}),
                               ('superres', {'resize_ratio': 0.5}),
                               ('superres', {'resize_ratio': 0.25}),
                               ('cs', {'compress_ratio': 0.1}),
                               ('videocs', {'compress_ratio': 0.1}),
                               ('blurdisk', {'size': 4,
                                             'radius': 2.}),
                               ('blurmotion', {'size': 7}),
                               ('videoblurdisk', {'size': 4,
                                                  'radius': 2.}),
                               ('frameinterp', {'interp_ratio': 0.5}),
                               ('frameinterp', {'interp_ratio': 0.25}),
                               ('prediction', {'predict_ratio': 0.75}),
                               ('prediction', {'predict_ratio': 0.5}),
                               ('prediction', {'predict_ratio': 0.25}),
                               ('colorization', {})]}
    _data['problems_train'] = _data['problems_test']
    _algo = {'batch_size': _data['batch_size'],
             'shape1': _data['frames'],
             'shape2': _data['size'],
             'shape3': _data['size'],
             'shape4': 1,
             'max_iter': 13,
             'filters': 64,
             'filter_size_enc': 3,
             'filter_size_dec': 11,
             'rnn': True,
             'l2': 0.,
             'epochs': 50,
             'patience': 5,
             'lr': 1e-4,
             'clipnorm': 1.,
             'dtype': _data['dtype'],
             'workers': 14,
             'max_queue_size': 10,
             'gpus': 1}

             
@ex.named_config
def videoonenet():
    _algo = {'mode': 'videoonenet',
             'rho': 0.3}

@ex.named_config
def videoonenetadmm():
    _algo = {'mode': 'videoonenetadmm',
             'rho': 0.3}

@ex.named_config
def videowaveletsparsityadmm():
    _algo = {'mode': 'videowaveletsparsityadmm',
             'rho': 0.3,
             'lambda_l1': 0.05}

@ex.named_config
def rnn():
    _algo = {'rnn': True}

@ex.named_config
def nornn():
    _algo = {'rnn': False}


@ex.automain
def run(_data, _algo, _rnd, _seed):
    _data_train = _data.copy()
    _data_train['problems'] = _data['problems_train']
    data = MNISTRotatedDataset(config=_data_train, seed=_seed)

    if _algo['mode'] == 'videoonenet':
        alg = KerasVideoOneNet(results_dir=results_dir, config=_algo)
    elif _algo['mode'] == 'videoonenetadmm':
        alg = KerasVideoOneNetADMM(results_dir=results_dir, config=_algo)
    elif _algo['mode'] == 'videowaveletsparsityadmm':
        alg = NumpyVideoWaveletSparsityADMM(results_dir=results_dir, config=_algo)
    alg.build()

    result = []
    
    alg.train(data.seq_train, X_val=data.seq_val)
    
    # remove threading from generator for reproducibility in testing 
    _algo_problem = _algo.copy()
    _algo_problem['workers'] = 1
    _algo_problem['max_queue_size'] = 1
    alg.set_config(_algo_problem)
    
    for problem in _data['problems_test']:
        _data_test = _data.copy()
        _data_test['problems'] = [problem]
        
	# generate some test images

        data.generate_sequences(config=_data_test, seed=_seed)
        for batch in range(4):
            alg.plot_predictions(data.seq_test[batch], problem, filepath=(results_dir + '/videos_test_%04d.png') % batch)
        
	# evaluate mean loss on test images
        data.generate_sequences(config=_data_test, seed=_seed)
        result_test = alg.test(data.seq_test)
        result.append([problem[0], problem[1], result_test])
        print(result[-1])

    return result

