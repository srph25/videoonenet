import tensorflow as tf
from keras import backend as K
import numpy as np
from scipy import linalg
import pywt


def kron(mat1, mat2):
    #Computes the Kronecker product of two matrices.
    mat1_rsh = K.reshape(mat1, [K.shape(mat1)[0], 1, K.shape(mat1)[1], 1])
    mat2_rsh = K.reshape(mat2, [1, K.shape(mat2)[0], 1, K.shape(mat2)[2]])
    return K.reshape(mat1_rsh * mat2_rsh, [K.shape(mat1)[0] * K.shape(mat2)[0], K.shape(mat1)[1] * K.shape(mat2)[1]])

def elu_like(x):
    return tf.nn.softplus(2. * x + 2.) / 2. - 1.

def batch_pinv(A, gpu=True, svd=False):
    if gpu is False:
        if svd is True:
            p = tf.numpy_function(lambda X: np.linalg.pinv(X, rcond=K.epsilon()), [A], K.floatx())
        else:
            p = tf.numpy_function(lambda X: np.concatenate([linalg.pinv(x, check_finite=False)[None, :, :] for x in X], axis=0), [A], K.floatx())
        A_shape = K.int_shape(A)
        p.set_shape(tf.TensorShape([A_shape[0], A_shape[2], A_shape[1]]))
    else:
        if svd is True:
            s, u, v = tf.svd(A)
            threshold = K.max(s, axis=-1, keepdims=True) * 1e-5
            s_inv = (1. / s) * K.cast(s > threshold, K.floatx())
            s_inv = tf.linalg.diag(s_inv)
            p = K.batch_dot(v, K.batch_dot(s_inv, K.permute_dimensions(u, (0, 2, 1)), axes=(2, 1)), axes=(2, 1))
        else:
            B = tf.linalg.diag(K.ones_like(A[:, :, 0]))
            p = tf.linalg.lstsq(A, B, l2_regularizer=1e-5)
    return p

def wavelet_transform(img, filters=None, levels=None):
    if levels is None:
        vimg = tf.pad(img, [(0, 0),
                            (0, 2 ** int(np.ceil(np.log2(K.int_shape(img)[1]))) - K.int_shape(img)[1]),
                            (0, 2 ** int(np.ceil(np.log2(K.int_shape(img)[2]))) - K.int_shape(img)[2]),
                            (0, 2 ** int(np.ceil(np.log2(K.int_shape(img)[3]))) - K.int_shape(img)[3]),
                            (0, 0)])
    else:
        vimg = img

    if filters is None:
        w = pywt.Wavelet('db4')
        dec_hi = np.array(w.dec_hi[::-1])
        dec_lo = np.array(w.dec_lo[::-1])
        filters = np.stack([dec_lo[None, None, :] * dec_lo[None, :, None] * dec_lo[:, None, None],
                            dec_lo[None, None, :] * dec_lo[None, :, None] * dec_hi[:, None, None],
                            dec_lo[None, None, :] * dec_hi[None, :, None] * dec_lo[:, None, None],
                            dec_lo[None, None, :] * dec_hi[None, :, None] * dec_hi[:, None, None],
                            dec_hi[None, None, :] * dec_lo[None, :, None] * dec_lo[:, None, None],
                            dec_hi[None, None, :] * dec_lo[None, :, None] * dec_hi[:, None, None],
                            dec_hi[None, None, :] * dec_hi[None, :, None] * dec_lo[:, None, None],
                            dec_hi[None, None, :] * dec_hi[None, :, None] * dec_hi[:, None, None]]).transpose((1, 2, 3, 0))[:, :, :, None, :]
        filters = K.constant(filters)
    if levels is None:
        print(K.int_shape(vimg)[1:4])
        levels = pywt.dwtn_max_level(K.int_shape(vimg)[1:4], 'db4')
        print(levels)

    t = vimg.shape[1]
    h = vimg.shape[2]
    w = vimg.shape[3]
    res = K.conv3d(vimg, filters, strides=(2, 2, 2), padding='same')
    if levels > 1:
        res = K.concatenate([wavelet_transform(res[:, :, :, :, :1], filters, levels=(levels - 1)), res[:, :, :, :, 1:]], axis=-1)
    '''
    res = K.permute_dimensions(res, (0, 4, 1, 2, 3))
    res = K.reshape(res, (-1, 2, t // 2, h // 2, w // 2))
    res = K.permute_dimensions(res, (0, 2, 1, 3, 4))
    res = K.reshape(res, (-1, 1, t, h, w))
    res = K.permute_dimensions(res, (0, 2, 3, 4, 1))
    '''
    res = K.reshape(res, (-1, t, h, w, 1))
    #print('wt', levels, K.int_shape(img), K.int_shape(vimg), K.int_shape(filters), K.int_shape(res))
    return res

def inverse_wavelet_transform(vres, inv_filters=None, output_shape=None, levels=None):
    if inv_filters is None:
        w = pywt.Wavelet('db4')
        rec_hi = np.array(w.rec_hi)
        rec_lo = np.array(w.rec_lo)
        inv_filters = np.stack([rec_lo[None, None, :] * rec_lo[None, :, None] * rec_lo[:, None, None],
                                rec_lo[None, None, :] * rec_lo[None, :, None] * rec_hi[:, None, None],
                                rec_lo[None, None, :] * rec_hi[None, :, None] * rec_lo[:, None, None],
                                rec_lo[None, None, :] * rec_hi[None, :, None] * rec_hi[:, None, None],
                                rec_hi[None, None, :] * rec_lo[None, :, None] * rec_lo[:, None, None],
                                rec_hi[None, None, :] * rec_lo[None, :, None] * rec_hi[:, None, None],
                                rec_hi[None, None, :] * rec_hi[None, :, None] * rec_lo[:, None, None],
                                rec_hi[None, None, :] * rec_hi[None, :, None] * rec_hi[:, None, None]]).transpose((1, 2, 3, 0))[:, :, :, None, :]
        inv_filters = K.constant(inv_filters)
    if levels is None:
        levels = pywt.dwtn_max_level(K.int_shape(vres)[1:4], 'db4')
        print(levels)

    t = vres.shape[1]
    h = vres.shape[2]
    w = vres.shape[3]
    '''
    res = K.permute_dimensions(vres, (0, 4, 1, 2, 3))
    res = K.reshape(res, (-1, t // 2, 2, h // 2, w // 2))
    res = K.permute_dimensions(res, (0, 2, 1, 3, 4))
    res = K.reshape(res, (-1, 8, t // 2, h // 2, w // 2))
    res = K.permute_dimensions(res, (0, 2, 3, 4, 1))
    '''
    res = K.reshape(vres, (-1, t // 2, h // 2, w // 2, 8))
    if levels > 1:
        res = K.concatenate([inverse_wavelet_transform(res[:, :, :, :, :1], inv_filters, output_shape=(K.shape(vres)[0], K.shape(vres)[1] // 2, K.shape(vres)[2] // 2, K.shape(vres)[3] // 2, K.shape(vres)[4]), levels=(levels - 1)), res[:, :, :, :, 1:]], axis=-1)
    res = K.conv3d_transpose(res, inv_filters, output_shape=K.shape(vres), strides=(2, 2, 2), padding='same')
    
    out = res[:, :output_shape[1], :output_shape[2], :output_shape[3], :]
    #print('iwt', levels, K.int_shape(vres), K.int_shape(inv_filters), K.int_shape(res), K.int_shape(out), output_shape)
    return out

def soft_thresholding(x, beta):
    y = K.relu(x - beta) - K.relu(-x - beta)
    return y

