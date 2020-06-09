import os
import string
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.objectives import mean_squared_error
from keras.regularizers import l2
from keras.callbacks import TerminateOnNaN, ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.utils import multi_gpu_model, Sequence
from utils.ops import *
from utils.plot import *
from utils.utils import *
from utils.layers import *
from PIL import Image, ImageDraw, ImageFont

def psnr(true, pred):
    return 10. * K.log(K.cast(K.prod(K.shape(true)[1:]), K.floatx()) / K.sum(K.square(pred - true), axis=[1, 2, 3, 4])) / K.log(10.)

def psnr_masked(true, pred):
    mask = K.cast(K.not_equal(true, self.config['pad_value']), K.floatx())
    return 10. * K.log(K.cast(K.sum(1 - K.all(1 - mask, axis=[2, 3, 4]), axis=1) * K.prod(K.shape(true)[2:]), K.floatx()) / K.sum(K.square(pred - true) * mask, axis=[1, 2, 3, 4])) / K.log(10.)
  
def var(xi, mean):
    return K.square(xi - mean)

class KerasVideoOneNet():
    def __init__(self, results_dir, config):
        self.results_dir = results_dir
        self.set_config(config)
        if 'pad_value' in self.config.keys():
            self.loss = lambda true, pred: K.mean(K.square(pred - true) * K.cast(K.not_equal(true, self.config['pad_value']), K.floatx()), axis=-1)
            self.psnr = psnr_masked
        else:
            self.loss = lambda true, pred: K.mean(K.square(pred - true), axis=-1)
            if 'mean' not in self.config.keys():
                self.psnr = psnr
            else:
                self.psnr = lambda true, pred: var(psnr(true, pred), self.config['mean'])

    def set_config(self, config):
        assert(np.mod(config['batch_size'], config['gpus']) == 0)
        self.config = config

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
        init_out_z = inp_pinvA_y

        # u init
        init_lmbd = Lambda(lambda z: K.zeros_like(z))
        init_out_u = init_lmbd(init_out_z)

        self.init = Model([inp_y, inp_A, inp_AT_y, inp_pinvA_y, inp_pinvB], [init_out_z, init_out_u])

    def build_prior(self):
        inp_z = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']))
        inp_u = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']))
        
        # x update
        prior_concat = Lambda(lambda lst: K.concatenate(lst, axis=-1))
        prior_concat_out = prior_concat([inp_z, inp_u])

        enc_conv1 = Conv2D(self.config['filters'], self.config['filter_size_enc'], padding='same', kernel_regularizer=l2(self.config['l2']))
        enc_tdconv1 = TimeDistributed(enc_conv1)
        enc_tdconv1_out = enc_tdconv1(prior_concat_out)
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
            enc_crnn1 = Bidirectional(ConvMinimalRNN2D(self.config['filters'], self.config['filter_size_enc'], recurrent_activation='sigmoid', padding='same', return_sequences=True, unroll=True, kernel_regularizer=l2(self.config['l2'])))
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

        _data_lmbd1 = Lambda(lambda lst: lst[0] - lst[1]) # x - u
        _data_lmbd1_out = _data_lmbd1([inp_x, inp_u])
        
        _data_concat = Lambda(lambda lst: K.concatenate([lst[0], sqrt_rho * K.batch_flatten(lst[1])], axis=-1))
        _data_concat_out = _data_concat([inp_y, _data_lmbd1_out])
        _data_batch_dot = Lambda(lambda lst: K.batch_dot(lst[0], lst[1], axes=(2, 1)))
        _data_batch_dot_out = _data_batch_dot([inp_pinvB, _data_concat_out])

        _data_rshp = Reshape((self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']))
        _data_out_z = _data_rshp(_data_batch_dot_out)

        # u update
        _data_lmbd2 = Lambda(lambda lst: lst[2] + (lst[1] - lst[0])) # u + (z - x)
        _data_out_u = _data_lmbd2([inp_x, _data_out_z, inp_u])

        # concat
        data_concat = Lambda(lambda lst: K.concatenate(lst, axis=-1))
        data_concat_out = data_concat([inp_AT_y, inp_pinvA_y, inp_x, inp_z, inp_u, _data_out_z, _data_out_u])

        enc_conv1 = Conv2D(self.config['filters'], self.config['filter_size_enc'], padding='same', kernel_regularizer=l2(self.config['l2']))
        enc_tdconv1 = TimeDistributed(enc_conv1)
        enc_tdconv1_out = enc_tdconv1(data_concat_out)
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
            enc_crnn1 = Bidirectional(ConvMinimalRNN2D(self.config['filters'], self.config['filter_size_enc'], recurrent_activation='sigmoid', padding='same', return_sequences=True, unroll=True, kernel_regularizer=l2(self.config['l2'])))
            enc_crnn1_out = enc_crnn1(enc_act3_out)
            enc_out = enc_crnn1_out
        
        dec_conv1 = Conv2DTranspose(2 * self.config['shape4'], self.config['filter_size_dec'], activation='linear', padding='same', kernel_regularizer=l2(self.config['l2']))
        dec_tdconv1 = TimeDistributed(dec_conv1)
        dec_tdconv1_out = dec_tdconv1(enc_out)

        data_out_z_u = dec_tdconv1_out

        # u update
        data_lmbd1 = Lambda(lambda z_u: z_u[:, :, :, :, :self.config['shape4']])
        data_out_z = data_lmbd1(data_out_z_u)
        
        data_lmbd2 = Lambda(lambda z_u: z_u[:, :, :, :, self.config['shape4']:])
        data_out_u = data_lmbd2(data_out_z_u)
        
        self.data = Model([inp_y, inp_A, inp_AT_y, inp_pinvA_y, inp_pinvB, inp_x, inp_z, inp_u], [data_out_z, data_out_u])

    def compile(self, inputs, outputs, optimizer=None, loss=None, metrics=None, loss_weights=None, dtype='float32'):
        if self.config['gpus'] > 1:
            with tf.device('/cpu:0'):
                model_base = Model(inputs, outputs)
            model = multi_gpu_model(model_base, gpus=self.config['gpus'])
        else:
            model_base = Model(inputs, outputs)
            model = model_base
        model_base.summary()
        if optimizer is None:
            optimizer = Adam(amsgrad=False, lr=self.config['lr'], clipnorm=self.config['clipnorm'])
        if dtype == 'float16':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config['lr'])
            optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
            optimizer = TFOptimizer(optimizer)
        if loss is None:
            loss = mean_squared_error
        if metrics is None:
            metrics = []
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics,
                      loss_weights=loss_weights)
        return model, model_base

    def build(self):
        K.clear_session()

        self.build_precomp()
        self.build_init()
        self.build_prior()
        self.build_data()
        
        inp_y = Input(shape=(None,))
        inp_A = Input(shape=(None, self.config['shape1'] * self.config['shape2'] * self.config['shape3'] * self.config['shape4']))
        admm_out = UnrolledOptimization(model_precomp=self.precomp, model_init=self.init, model_prior=self.prior, model_data=self.data, max_iter=self.config['max_iter'], shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']), unroll=True)([inp_y, inp_A])
        self.admm, self.admm_base = self.compile([inp_y, inp_A], admm_out, loss=self.loss, metrics=[self.psnr], dtype=self.config['dtype'])

    def train(self, X_train, X_val=None):
        assert(isinstance(X_train, Sequence) or isinstance(X_train, tuple))
        if X_val is not None:
            assert(isinstance(X_val, Sequence) or isinstance(X_val, tuple))
        hdf5 = self.results_dir + '/admm.hdf5'
        if os.path.exists(hdf5):
            os.remove(hdf5)
        model_checkpoint = ModelCheckpoint(filepath=hdf5, save_best_only=True if (X_val is not None) else False, save_weights_only=True)
        early_stopping = EarlyStopping(patience=self.config['patience'])
        terminate_on_nan = TerminateOnNaN()
        if isinstance(X_train, tuple):
            self.admm.fit(X_train[0], X_train[1],
                          batch_size=self.config['batch_size'],
                          epochs=self.config['epochs'],
                          verbose=1,
                          callbacks=[model_checkpoint, early_stopping, terminate_on_nan] if (X_val is not None) else [model_checkpoint, terminate_on_nan],
                          validation_data=X_val if X_val is not None else None)
        elif isinstance(X_train, Sequence):
            self.admm.fit_generator(X_train,
                                    epochs=self.config['epochs'],
                                    verbose=1,
                                    callbacks=[model_checkpoint, early_stopping, terminate_on_nan] if (X_val is not None) else [model_checkpoint, terminate_on_nan],
                                    validation_data=X_val if X_val is not None else None,
                                    shuffle=True, workers=self.config['workers'], max_queue_size=self.config['max_queue_size'])
        self.admm.load_weights(hdf5)
        self.admm_base.save_weights(hdf5)
        #self.plot_decoder_weights()
    
    def test(self, X):
        if isinstance(X, tuple):
            return self.admm.evaluate(X[0], X[1], batch_size=self.config['batch_size'])
        elif isinstance(X, Sequence):
            return self.admm.evaluate_generator(X, workers=self.config['workers'], max_queue_size=self.config['max_queue_size'])

    def predict(self, X):
        if isinstance(X, tuple):
            return self.admm.predict(X[0], batch_size=self.config['batch_size'])
        elif isinstance(X, list):
            return self.admm.predict(X, batch_size=self.config['batch_size'])
        elif isinstance(X, Sequence):
            return self.admm.predict_generator(X, workers=self.config['workers'], max_queue_size=self.config['max_queue_size'])

    def plot_weights(self, weights, filepath=None, title='Weights'):
        if filepath is None:
            filepath = self.results_dir + '/weights.png'
        plot(filepath, title, make_mosaic(weights, nrows=4, ncols=self.config['filters'] // 4))
        
    def plot_decoder_weights(self):
        self.plot_weights(np.transpose(self.prior.get_weights()[-2], (3, 0, 1, 2)), filepath=self.results_dir + '/decoder_weights.png', title='Decoder Weights') # -4

    def plot_predictions(self, X, problem, filepath=None, title=None):
        if filepath is None:
            filepath = self.results_dir + '/videos.png'
        str_problem = str(problem).translate(str.maketrans(' ','_',string.punctuation))
        shp = (1,) + X[1][0].shape
        _, shp_out = eval(problem[0])(shp, **(problem[1]))
        for i in range(X[0][0].shape[0]):
            true = X[1][i].copy()
            plot((filepath[:-4] + '_%s_%04d_true' + filepath[-4:]) % (str_problem, i), title, make_mosaic(true, nrows=1, ncols=X[1].shape[1], clip=True))
            del true

            if len(shp_out) == 5:
                inp = np.reshape(X[0][0][i], shp_out[1:])
                plot((filepath[:-4] + '_%s_%04d_inp' + filepath[-4:]) % (str_problem, i), title, make_mosaic(inp, nrows=1, ncols=shp_out[1], clip=True))
                del inp

            pred = self.predict([X[0][0][i][None], X[0][1][i][None]])[0]
            psnr = self.test(([X[0][0][i][None], X[0][1][i][None]], X[1][i][None]))[1]
            mosaic = np.repeat(np.repeat(np.repeat(make_mosaic(pred, nrows=1, ncols=X[1].shape[1], clip=True), 2, axis=0), 2, axis=1), 3 if (self.config['shape4'] == 1) else 1, axis=2)
            img = ((mosaic * 255).astype(np.uint8))
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(font='utils/tnr.ttf', size=27 if shp[2]==32 else (18 if shp[2]==16 else 12))
            draw.text((0, 0), "%.2f" % psnr, (255, 0, 0), font=font)
            mosaic = np.array(img) / 255.
            plot((filepath[:-4] + '_%s_%04d_pred' + filepath[-4:]) % (str_problem, i), title, mosaic)
            del pred, mosaic
        #del X

