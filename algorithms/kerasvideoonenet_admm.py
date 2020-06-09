import os
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.objectives import mean_squared_error
from keras.regularizers import l2
from keras.initializers import Constant
from keras.callbacks import TerminateOnNaN, ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.utils import multi_gpu_model, Sequence
from utils.ops import *
from utils.plot import *
from utils.utils import *
from utils.layers import *
from algorithms.kerasvideoonenet import KerasVideoOneNet

class KerasVideoOneNetADMM(KerasVideoOneNet):
    def __init__(self, results_dir, config):
        super(KerasVideoOneNetADMM, self).__init__(results_dir, config)

    def build_precomp(self):
        inp_y = Input(shape=(None,))
        inp_A = Input(shape=(None, self.config['shape1'] * self.config['shape2'] * self.config['shape3'] * self.config['shape4']))

        # A.T y
        precomp_lmbd1 = Lambda(lambda lst: K.batch_dot(lst[0], K.permute_dimensions(lst[1], (0, 2, 1)), axes=(1, 2)))
        precomp_lmbd1_out = precomp_lmbd1([inp_y, inp_A])
        precomp_rshp1 = Reshape((self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']))
        precomp_AT_y = precomp_rshp1(precomp_lmbd1_out)

        # pinv(A) y
        precomp_lmbd2 = Lambda(lambda lst: K.batch_dot(lst[0], batch_pinv(lst[1], gpu=True), axes=(1, 2)))
        precomp_lmbd2_out = precomp_lmbd2([inp_y, inp_A])
        precomp_rshp2 = Reshape((self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']))
        precomp_pinvA_y = precomp_rshp2(precomp_lmbd2_out)

        # pinv(concat)
        sqrt_rho = np.sqrt(self.config['rho'])
        precomp_lmbd3 = Lambda(lambda A: batch_pinv(K.concatenate([A, sqrt_rho * tf.linalg.diag(K.ones_like(A[:, 0, :]))], axis=1), gpu=True))
        precomp_pinvB = precomp_lmbd3(inp_A)

        self.precomp = Model([inp_y, inp_A], [precomp_AT_y, precomp_pinvA_y, precomp_pinvB])

    def build_init(self):
        inp_y = Input(shape=(None,))
        inp_A = Input(shape=(None, self.config['shape1'] * self.config['shape2'] * self.config['shape3'] * self.config['shape4']))
        inp_AT_y = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']))
        inp_pinvA_y = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']))
        inp_pinvB = Input(shape=(self.config['shape1'] * self.config['shape2'] * self.config['shape3'] * self.config['shape4'], None))

        # z init
        init_out_z = inp_AT_y

        # u init
        init_lmbd = Lambda(lambda z: K.zeros_like(z))
        init_out_u = init_lmbd(init_out_z)

        self.init = Model([inp_y, inp_A, inp_AT_y, inp_pinvA_y, inp_pinvB], [init_out_z, init_out_u])

    def build_prior(self):
        inp_z = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']))
        inp_u = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']))
        
        # x update
        prior_lmbd = Lambda(lambda lst: lst[0] + lst[1]) # z + u
        prior_lmbd_out = prior_lmbd([inp_z, inp_u])

        enc_conv1 = Conv2D(self.config['filters'], self.config['filter_size_enc'], padding='same', kernel_regularizer=l2(self.config['l2']))
        enc_tdconv1 = TimeDistributed(enc_conv1)
        enc_tdconv1_out = enc_tdconv1(prior_lmbd_out)
        enc_norm1 = InstanceNormalization()
        enc_norm1_out = enc_norm1(enc_tdconv1_out)
        enc_act1 = Activation(lambda x: elu_like(x))
        enc_act1_out = enc_act1(enc_norm1_out)

        enc_conv2 = Conv2D(self.config['filters'], self.config['filter_size_enc'], padding='same', kernel_regularizer=l2(self.config['l2']))
        enc_tdconv2 = TimeDistributed(enc_conv2)
        enc_tdconv2_out = enc_tdconv2(enc_act1_out)
        enc_norm2 = InstanceNormalization()
        enc_norm2_out = enc_norm2(enc_tdconv2_out)
        enc_act2 = Activation(lambda x: elu_like(x))
        enc_act2_out = enc_act2(enc_norm2_out)
        
        enc_conv3 = Conv2D(self.config['filters'], self.config['filter_size_enc'], padding='same', kernel_regularizer=l2(self.config['l2']))
        enc_tdconv3 = TimeDistributed(enc_conv3)
        enc_tdconv3_out = enc_tdconv3(enc_act2_out)
        enc_norm3 = InstanceNormalization()
        enc_norm3_out = enc_norm3(enc_tdconv3_out)
        enc_act3 = Activation(lambda x: elu_like(x))
        enc_act3_out = enc_act3(enc_norm3_out)

        if self.config['rnn'] is False:
            enc_out = enc_act3_out
        else:
            enc_crnn1 = ConvMinimalRNN2D(self.config['filters'], self.config['filter_size_enc'], recurrent_activation='sigmoid', padding='same', return_sequences=True, unroll=True, kernel_regularizer=l2(self.config['l2']))
            enc_crnn1_out = enc_crnn1(enc_act3_out)
            enc_out = enc_crnn1_out
        
        dec_conv1 = Conv2DTranspose(self.config['shape4'], self.config['filter_size_dec'], activation='linear', padding='same', kernel_regularizer=l2(self.config['l2']))
        dec_tdconv1 = TimeDistributed(dec_conv1)
        dec_tdconv1_out = dec_tdconv1(enc_out)

        prior_dec_out = dec_tdconv1_out
        
        self.prior = Model([inp_z, inp_u], [prior_dec_out])

    def build_data(self):
        inp_y = Input(shape=(None,))
        inp_A = Input(shape=(None, self.config['shape1'] * self.config['shape2'] * self.config['shape3'] * self.config['shape4']))
        inp_AT_y = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']), name='AT_y')
        inp_pinvA_y = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']), name='pinvA_y')
        inp_pinvB = Input(shape=(self.config['shape1'] * self.config['shape2'] * self.config['shape3'] * self.config['shape4'], None), name='pinvB')
        inp_x = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']), name='x')
        inp_z = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']), name='z')
        inp_u = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']), name='u')

        # z update
        sqrt_rho = np.sqrt(self.config['rho'])

        data_lmbd1 = Lambda(lambda lst: lst[0] - lst[1]) # x - u
        data_lmbd1_out = data_lmbd1([inp_x, inp_u])
        
        data_concat = Lambda(lambda lst: K.concatenate([lst[0], sqrt_rho * K.batch_flatten(lst[1])], axis=-1))
        data_concat_out = data_concat([inp_y, data_lmbd1_out])
        data_batch_dot = Lambda(lambda lst: K.batch_dot(lst[0], lst[1], axes=(2, 1)))
        data_batch_dot_out = data_batch_dot([inp_pinvB, data_concat_out])
        
        data_rshp = Reshape((self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']))
        data_out_z = data_rshp(data_batch_dot_out)

        # u update
        data_lmbd2 = Lambda(lambda lst: lst[2] + (lst[1] - lst[0])) # u + (z - x)
        data_out_u = data_lmbd2([inp_x, data_out_z, inp_u])
        
        self.data = Model([inp_y, inp_A, inp_AT_y, inp_pinvA_y, inp_pinvB, inp_x, inp_z, inp_u], [data_out_z, data_out_u])

