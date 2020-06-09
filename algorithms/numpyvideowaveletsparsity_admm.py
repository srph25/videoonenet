import os
import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator
import pywt
from keras.utils import Sequence, OrderedEnqueuer
import tqdm
from algorithms.kerasvideoonenet_admm import KerasVideoOneNetADMM
from utils.plot import *

def wavelet_transform(x):
    w_coeffs_rgb = []
    for i in range(x.shape[4]):
        w_coeffs_list = pywt.wavedecn(x[0, :, :, :, i], 'db4', level=None, mode='periodization')
        w_coeffs, coeff_slices = pywt.coeffs_to_array(w_coeffs_list)
        w_coeffs_rgb.append(w_coeffs)

    w_coeffs_rgb = np.array(w_coeffs_rgb)
    return w_coeffs_rgb, coeff_slices

def inverse_wavelet_transform(w_coeffs_rgb, coeff_slices, x_shape):
    x_hat = np.zeros(x_shape)
    for i in range(w_coeffs_rgb.shape[0]):
        w_coeffs_list = pywt.array_to_coeffs(w_coeffs_rgb[i, :, :, :], coeff_slices)
        x_hat[0, :, :, :, i] = pywt.waverecn(w_coeffs_list, wavelet='db4', mode='periodization')
    return x_hat

def soft_threshold(x, beta):
    y = np.maximum(0, x - beta) - np.maximum(0, -x - beta)
    return y

def solve(y, A, x_shape, lambda_l1=0.05, rho=0.3, max_iter=300, solver_tol=1e-6):
    """ See Wang, Yu, Wotao Yin, and Jinshan Zeng. "Global convergence of ADMM in nonconvex nonsmooth optimization."
    arXiv preprint arXiv:1511.06324 (2015).
    It provides convergence condition: basically with large enough alpha, the program will converge. """
    def A_fun(x):
        y = np.dot(A, x.ravel()) # dot product with A
        y = np.reshape(y, (1, -1))
        return y

    def AT_fun(y):
        y = np.reshape(y, (-1, 1))
        x = np.dot(A.T, y) # dot product with A_T
        x = np.reshape(x, x_shape)
        return x

    ATy = AT_fun(y)
    x_shape = ATy.shape
    d = np.prod(x_shape)

    def A_cgs_fun(x): # Q
        x = np.reshape(x, x_shape)
        y = AT_fun(A_fun(x)) + rho * x
        return y.ravel()
    A_cgs = LinearOperator((d, d), matvec=A_cgs_fun, dtype='float') # Q

    def compute_p_inv_A(b, z0):
        (z,info) = scipy.sparse.linalg.cgs(A_cgs, b.ravel(), x0=z0.ravel(), tol=1e-3, maxiter=100) # inv(Q) b
        if info > 0:
            print('cgs convergence to tolerance not achieved')
        elif info <0:
            print('cgs gets illegal input or breakdown')
        z = np.reshape(z, x_shape)
        return z

    def A_cgs_fun_init(x): # A_T_A
        x = np.reshape(x, x_shape)
        y = AT_fun(A_fun(x))
        return y.ravel()
    A_cgs_init = LinearOperator((d,d), matvec=A_cgs_fun_init, dtype='float') # A_T_A

    def compute_init(b, z0):
        (z,info) = scipy.sparse.linalg.cgs(A_cgs_init, b.ravel(), x0=z0.ravel(), tol=1e-2) # inv(A_T_A) b
        if info > 0:
            print('cgs convergence to tolerance not achieved')
        elif info <0:
            print('cgs gets illegal input or breakdown')
        z = np.reshape(z, x_shape)
        return z

    # initialize z and u
    z = compute_init(ATy, ATy) # inv(A_T_A) AT y=pinv(A) y
    u = np.zeros(x_shape)

    for _ in range(max_iter):
        # this implementation has x and z swapped compared to reference code from Boyd... but is no problem, just naming

        # x-update
        # z_hat = np.sign(u) * np.maximum(0, np.abs(u) - l_over_rho)
        # but in wavelet domain...
        net_input = z + u
        Wzu, wbook = wavelet_transform(net_input) # W (z+u)
        q = soft_threshold(Wzu, lambda_l1 / (rho + 1e-7)) # softthres(W (z+u))
        x = inverse_wavelet_transform(q, wbook, x_shape) # invW softthres(W (z+u))
        x = np.reshape(x, x_shape)

        # z-update
        # np.dot(Q, A_t_y + rho*(z_hat - u))
        b = ATy + rho * (x - u)
        z = compute_p_inv_A(b, z)

        # u-update
        # u = x_hat + u
        # u = u - z_hat
        u += z - x; 

        x_z = np.sqrt(np.mean(np.square(x - z)))

        if x_z < solver_tol:
            break

    return z

def weighted(fn, y_true, y_pred, weights=None, mask=None):
    """Wrapper function.
    # Arguments
        y_true: `y_true` argument of `fn`.
        y_pred: `y_pred` argument of `fn`.
        weights: Weights tensor.
        mask: Mask tensor.
    # Returns
        Scalar tensor.
    """
    # score_array has ndim >= 2
    score_array = fn(y_true, y_pred)
    if mask is not None:
        # Cast the mask to floatX to avoid float64 upcasting in Theano
        mask = mask.astype(np.float32)
        # mask should have the same shape as score_array
        score_array *= mask
        #  the loss per batch should be proportional
        #  to the number of unmasked samples.
        score_array /= np.mean(mask) + 1e-7

    # apply sample weighting
    if weights is not None:
        # reduce score_array to same ndim as weight array
        ndim = score_array.ndim
        weight_ndim = weights.ndim
        score_array = np.mean(score_array,
                              axis=list(range(weight_ndim, ndim)))
        score_array *= weights
        score_array /= np.mean((weights != 0).astype(np.float32))
    return np.mean(score_array)

def psnr(true, pred):
    return 10 * np.log(np.prod(true.shape[1:]) / np.sum(np.square(pred - true), axis=(1, 2, 3, 4))) / np.log(10)

def psnr_masked(true, pred):
    mask = (true != self.config['pad_value']).astype(np.float32)
    return 10 * np.log(np.sum(1 - np.all(1 - mask, axis=[2, 3, 4]), axis=1) * np.prod(true.shape[2:]) / np.sum(np.square(pred - true) * mask, axis=[1, 2, 3, 4])) / np.log(10)

def var(xi, mean):
    return np.square(xi - mean)

class NumpyVideoWaveletSparsityADMM(KerasVideoOneNetADMM):
    def __init__(self, results_dir, config):
        super(NumpyVideoWaveletSparsityADMM, self).__init__(results_dir, config)
        if 'pad_value' in self.config.keys():
            self.loss = lambda true, pred: np.mean(np.square(pred - true) * (true != self.config['pad_value']).astype(np.float32), axis=-1)
            self.psnr = psnr_masked
        else:
            self.loss = lambda true, pred: np.mean(np.square(pred - true), axis=-1)
            if 'mean' not in self.config.keys():
                self.psnr = psnr
            else:
                self.psnr = lambda true, pred: var(psnr(true, pred), self.config['mean'])

    def build(self):
        pass
        
    def train(self, X_train, X_val=None):
        pass
        
    def test(self, X):
        l = 0.
        p = 0.
        num_samples = 0
        if isinstance(X, tuple):
            y, A = X[0]
            x = X[1]
            for batch_start in tqdm.tqdm(range(0, y.shape[0], self.config['batch_size'])):
                batch_end = np.min([batch_start + self.config['batch_size'], y.shape[0]])
                y_batch, A_batch, x_batch = y[batch_start:batch_end], A[batch_start:batch_end], x[batch_start:batch_end]
                batch = ([y_batch, A_batch], x_batch)
                pred_batch = []
                for i in range(batch_start, batch_end):
                    pred = solve(y_batch[i], A_batch[i], (1, self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']), rho=self.config['rho'], max_iter=self.config['max_iter'])
                    pred_batch.append(pred)
                pred_batch = np.concatenate(pred_batch, axis=0)
                loss = weighted(self.loss, x_batch, pred_batch)
                psnr = weighted(self.psnr, x_batch, pred_batch)
                l += (y_batch.shape[0] * loss)
                p += (y_batch.shape[0] * psnr)
                num_samples += y_batch.shape[0]
        elif isinstance(X, Sequence):
            if self.config['workers'] > 0:
                enqueuer = OrderedEnqueuer(X, use_multiprocessing=False)
                enqueuer.start(workers=self.config['workers'], max_queue_size=self.config['max_queue_size'])
                output_generator = enqueuer.get()
            else:
                output_generator = X
            l = 0.
            for steps_done in tqdm.tqdm(range(len(X))):
                generator_output = next(output_generator)
                (y_batch, A_batch), x_batch = generator_output
                pred_batch = []
                for i in range(0, y_batch.shape[0]):
                    pred = solve(y_batch[i], A_batch[i], (1, self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']), rho=self.config['rho'], max_iter=self.config['max_iter'])
                    pred_batch.append(pred)
                pred_batch = np.concatenate(pred_batch, axis=0)
                loss = weighted(self.loss, x_batch, pred_batch)
                psnr = weighted(self.psnr, x_batch, pred_batch)
                l += (y_batch.shape[0] * loss)
                p += (y_batch.shape[0] * psnr)
                num_samples += y_batch.shape[0]
        return [(l / num_samples), (p / num_samples)]

    def predict(self, X):
        outputs = []
        if isinstance(X, tuple) or isinstance(X, list):
            if isinstance(X, tuple):
                y, A = X[0]
            elif isinstance(X, list):
                y, A = X
            for batch_start in tqdm.tqdm(range(0, y.shape[0], self.config['batch_size'])):
                batch_end = np.min([batch_start + self.config['batch_size'], y.shape[0]])
                y_batch, A_batch = y[batch_start:batch_end], A[batch_start:batch_end]
                pred_batch = []
                for i in range(batch_start, batch_end):
                    print(batch_start, i)
                    pred = solve(y_batch[i], A_batch[i], (1, self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']), rho=self.config['rho'], max_iter=self.config['max_iter'])
                    pred_batch.append(pred)
                pred_batch = np.concatenate(pred_batch, axis=0)
                outputs.append(pred_batch)
        elif isinstance(X, Sequence):
            if self.config['workers'] > 0:
                enqueuer = OrderedEnqueuer(X, use_multiprocessing=False)
                enqueuer.start(workers=self.config['workers'], max_queue_size=self.config['max_queue_size'])
                output_generator = enqueuer.get()
            else:
                output_generator = X
            for steps_done in tqdm.tqdm(range(len(X))):
                generator_output = next(output_generator)
                (y_batch, A_batch), _ = generator_output
                pred_batch = []
                for i in range(0, y_batch.shape[0]):
                    pred = solve(y_batch[i], A_batch[i], (1, self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']), rho=self.config['rho'], max_iter=self.config['max_iter'])
                    pred_batch.append(pred)
                pred_batch = np.concatenate(pred_batch, axis=0)
                outputs.append(pred_batch)
        return np.concatenate(outputs, axis=0)

