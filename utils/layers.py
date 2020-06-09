import numpy as np
from keras.layers import *
from keras.layers.convolutional_recurrent import ConvRNN2D
from keras.layers.wrappers import Wrapper
from keras.legacy import interfaces
from keras import initializers, regularizers, constraints, activations
from keras.utils.generic_utils import has_arg, object_list_uid
from keras.layers.merge import _Merge
from keras import backend as K
import tensorflow as tf
from keras.layers.recurrent import _generate_dropout_mask
from keras.utils import conv_utils


def __init__unroll(self, cell,
                   return_sequences=False,
                   return_state=False,
                   go_backwards=False,
                   stateful=False,
                   unroll=False,
                   **kwargs):
    if isinstance(cell, (list, tuple)):
        # The StackedConvRNN2DCells isn't implemented yet.
        raise TypeError('It is not possible at the moment to'
                        'stack convolutional cells.')
    super(ConvRNN2D, self).__init__(cell,
                                    return_sequences,
                                    return_state,
                                    go_backwards,
                                    stateful,
                                    unroll,
                                    **kwargs)
    self.input_spec = [InputSpec(ndim=5)]
ConvRNN2D.__init__ = __init__unroll

def get_initial_state(self, inputs):
    # (samples, timesteps, rows, cols, filters)
    initial_state = K.zeros_like(inputs)
    # (samples, rows, cols, filters)
    initial_state = K.sum(initial_state, axis=1)
    shape = list(self.cell.kernel_shape)
    shape[-1] = self.cell.filters
    initial_state = self.cell.input_conv(initial_state,
                                         # K.zeros(tuple(shape)), # does not work with Unrolled Optimization
                                         K.constant(np.zeros(tuple(shape))),
                                         padding=self.cell.padding)
    # Fix for Theano because it needs
    # K.int_shape to work in call() with initial_state.
    keras_shape = list(K.int_shape(inputs))
    keras_shape.pop(1)
    if K.image_data_format() == 'channels_first':
        indices = 2, 3
    else:
        indices = 1, 2
    for i, j in enumerate(indices):
        keras_shape[j] = conv_utils.conv_output_length(
            keras_shape[j],
            shape[i],
            padding=self.cell.padding,
            stride=self.cell.strides[i],
            dilation=self.cell.dilation_rate[i])
    initial_state._keras_shape = keras_shape

    if hasattr(self.cell.state_size, '__len__'):
        return [initial_state for _ in self.cell.state_size]
    else:
        return [initial_state]
ConvRNN2D.get_initial_state = get_initial_state



class ConvMinimalRNN2DCell(Layer):
    """Cell class for the ConvMinimalRNN2DCell layer.
    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the strides of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `"channels_last"` (default) or `"channels_first"`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(ConvMinimalRNN2DCell, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        #self.data_format = K.normalize_data_format(data_format)
        self.data_format = 'channels_last'
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = self.filters
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, input_dim)
        self.kernel_shape = kernel_shape
        recurrent_kernel_shape = self.kernel_size + (input_dim, input_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=1)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[0]),
                self.recurrent_dropout,
                training=training,
                count=1)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state

        if 0 < self.dropout < 1.:
            inputs = inputs * dp_mask[0]

        if 0 < self.recurrent_dropout < 1.:
            h_tm1 = h_tm1 * rec_dp_mask[0]

        u1 = self.input_conv(inputs, self.kernel, self.bias, padding=self.padding)
        u2 = self.recurrent_conv(h_tm1, self.recurrent_kernel)
        u = self.recurrent_activation(u1 + u2)

        h = (1 - u) * h_tm1 + u * inputs
        
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        return h, [h]

    def input_conv(self, x, w, b=None, padding='valid'):
        conv_out = K.conv2d(x, w, strides=self.strides,
                            padding=padding,
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate)
        if b is not None:
            conv_out = K.bias_add(conv_out, b,
                                  data_format=self.data_format)
        return conv_out

    def recurrent_conv(self, x, w):
        conv_out = K.conv2d(x, w, strides=(1, 1),
                            padding='same',
                            data_format=self.data_format)
        return conv_out

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvMinimalRNN2DCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvMinimalRNN2D(ConvRNN2D):
    """ConvMinimalRNN2D
    It is similar to a MinimalRNN layer, but the input transformations
    and recurrent transformations are both convolutional.
    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the strides of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `"channels_last"` (default) or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, time, ..., channels)`
            while `"channels_first"` corresponds to
            inputs with shape `(batch, time, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    # Input shape
        - if data_format='channels_first'
            5D tensor with shape:
            `(samples, time, channels, rows, cols)`
        - if data_format='channels_last'
            5D tensor with shape:
            `(samples, time, rows, cols, channels)`
    # Output shape
        - if `return_sequences`
             - if data_format='channels_first'
                5D tensor with shape:
                `(samples, time, filters, output_row, output_col)`
             - if data_format='channels_last'
                5D tensor with shape:
                `(samples, time, output_row, output_col, filters)`
        - else
            - if data_format='channels_first'
                4D tensor with shape:
                `(samples, filters, output_row, output_col)`
            - if data_format='channels_last'
                4D tensor with shape:
                `(samples, output_row, output_col, filters)`
            where o_row and o_col depend on the shape of the filter and
            the padding
    # Raises
        ValueError: in case of invalid constructor arguments.
    # References
        - [Convolutional LSTM Network: A Machine Learning Approach for
        Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1)
        The current implementation does not include the feedback loop on the
        cells output
    """

    @interfaces.legacy_convlstm2d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        #cell = ConvLSTM2DCell(filters=filters,
        cell = ConvMinimalRNN2DCell(filters=filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding=padding,
                              data_format=data_format,
                              dilation_rate=dilation_rate,
                              activation=activation,
                              recurrent_activation=recurrent_activation,
                              use_bias=use_bias,
                              kernel_initializer=kernel_initializer,
                              recurrent_initializer=recurrent_initializer,
                              bias_initializer=bias_initializer,
                              kernel_regularizer=kernel_regularizer,
                              recurrent_regularizer=recurrent_regularizer,
                              bias_regularizer=bias_regularizer,
                              kernel_constraint=kernel_constraint,
                              recurrent_constraint=recurrent_constraint,
                              bias_constraint=bias_constraint,
                              dropout=dropout,
                              recurrent_dropout=recurrent_dropout)
        super(ConvMinimalRNN2D, self).__init__(cell,
                                         return_sequences=return_sequences,
                                         go_backwards=go_backwards,
                                         stateful=stateful,
                                         **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    @property
    def filters(self):
        return self.cell.filters

    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_initial_state(self, inputs):
        # (samples, timesteps, rows, cols, filters)
        initial_state = K.zeros_like(inputs)
        # (samples, rows, cols, filters)
        initial_state = K.sum(initial_state, axis=1)

        # for some reason the original variant of this method's input_conv part here does not work with Unrolled Optimization
        # fortunately, it is not needed

        if hasattr(self.cell.state_size, '__len__'):
            return [initial_state for _ in self.cell.state_size]
        else:
            return [initial_state]

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvMinimalRNN2D, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class RegularizedLambda(Lambda):
    def __init__(self, function, output_shape=None, mask=None, arguments=None, activity_regularizer=None, **kwargs):
        super(RegularizedLambda, self).__init__(function, output_shape=None, mask=None, arguments=None, **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
    
    def get_config(self):
        config = {'activity_regularizer': regularizers.serialize(self.activity_regularizer)}
        base_config = super(RegularizedLambda, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class UnrolledOptimization(_Merge):
    def __init__(self, model_precomp, model_init, model_prior, model_data, max_iter, shape, unroll=False, **kwargs):
        self.model_precomp = model_precomp
        self.model_init = model_init
        self.model_prior = model_prior
        self.model_data = model_data
        self.max_iter = max_iter
        self.shape = shape
        self.unroll = unroll
        self.supports_masking = False # True may work but almost surely needs effort
        self._num_constants = None
        self._input_map = {}
        super(UnrolledOptimization, self).__init__(**kwargs)

    def build(self, input_shape):
        assert(len(input_shape) == 2)
        assert(len(input_shape[0]) == 2)
        assert(len(input_shape[1]) == 3)
        if not self.model_precomp.built:
            self.model_precomp.build(input_shape)
            self.model_precomp.built = True
        if not self.model_init.built:
            self.model_init.build(input_shape + self.model_precomp.compute_output_shape(input_shape))
            self.model_init.built = True
        if not self.model_prior.built:
            self.model_prior.build(self.model_init.compute_output_shape(input_shape + self.model_precomp.compute_output_shape(input_shape)))
            self.model_prior.built = True
        if not self.model_data.built:
            self.model_data.build(input_shape + self.model_precomp.compute_output_shape(input_shape) + self.model_prior.compute_output_shape(self.model_init.compute_output_shape(input_shape + self.model_precomp.compute_output_shape(input_shape))) + self.model_init.compute_output_shape(input_shape + self.model_precomp.compute_output_shape(input_shape)))
            self.model_data.built = True
        self._reshape_required = False
        self.built = True
 
    @property
    def activity_regularizer(self):
        lmbd = lambda x: 0
        cond = False
        if hasattr(self.model_precomp, 'activity_regularizer'):
            lmbd = lambda x: (lmbd(x) + self.model_precomp.activity_regularizer(x))
            cond = True
        if hasattr(self.model_init, 'activity_regularizer'):
            lmbd = lambda x: (lmbd(x) + self.model_init.activity_regularizer(x))
            cond = True
        if hasattr(self.model_prior, 'activity_regularizer'):
            lmbd = lambda x: (lmbd(x) + self.model_prior.activity_regularizer(x))
            cond = True
        if hasattr(self.model_data, 'activity_regularizer'):
            lmbd = lambda x: (lmbd(x) + self.model_data.activity_regularizer(x))
            cond = True
        if cond is True:
            return lmbd
        else:
            return None

    @property
    def trainable(self):
        return (self.model_precomp.trainable or self.model_init.trainable or self.model_prior.trainable or self.model_data.trainable)

    @trainable.setter
    def trainable(self, value):
        self.model_precomp.trainable = value
        self.model_init.trainable = value
        self.model_prior.trainable = value
        self.model_data.trainable = value

    @property
    def trainable_weights(self):
        return self.model_precomp.trainable_weights +self.model_init.trainable_weights + self.model_prior.trainable_weights + self.model_data.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.model_precomp.non_trainable_weights + self.model_init.non_trainable_weights + self.model_prior.non_trainable_weights + self.model_data.non_trainable_weights

    @property
    def updates(self):
        upd = []
        if hasattr(self.model_precomp, 'updates'):
            upd += self.model_precomp.updates
        if hasattr(self.model_init, 'updates'):
            upd += self.model_init.updates
        if hasattr(self.model_prior, 'updates'):
            upd += self.model_prior.updates
        if hasattr(self.model_data, 'updates'):
            upd += self.model_data.updates
        return upd

    def get_updates_for(self, inputs=None):
        # If the wrapper modifies the inputs, use the modified inputs to
        # get the updates from the inner layer.
        inner_inputs = inputs
        if inputs is not None:
            uid = object_list_uid(inputs)
            if uid in self._input_map:
                inner_inputs = self._input_map[uid]

        updates = self.model_precomp.get_updates_for(inner_inputs)
        updates += self.model_init.get_updates_for(inner_inputs)
        updates += self.model_prior.get_updates_for(inner_inputs)
        updates += self.model_data.get_updates_for(inner_inputs)
        updates += super(UnrolledOptimization, self).get_updates_for(inputs)
        return updates

    @property
    def losses(self):
        lss = []
        if hasattr(self.model_precomp, 'losses'):
            lss += self.model_precomp.losses
        if hasattr(self.model_init, 'losses'):
            lss += self.model_init.losses
        if hasattr(self.model_prior, 'losses'):
            lss += self.model_prior.losses
        if hasattr(self.model_data, 'losses'):
            lss += self.model_data.losses
        return lss

    def get_losses_for(self, inputs=None):
        if inputs is None:
            losses = self.model_precomp.get_losses_for(None)
            losses += self.model_init.get_losses_for(None)
            losses += self.model_prior.get_losses_for(None)
            losses += self.model_data.get_losses_for(None)
            return losses + super(UnrolledOptimization, self).get_losses_for(None)
        return super(UnrolledOptimization, self).get_losses_for(inputs)

    def get_weights(self):
        return self.model_precomp.get_weights() + self.model_init.get_weights() + self.model_prior.get_weights() + self.model_data.get_weights()

    def set_weights(self, weights):
        self.model_precomp.set_weights(weights[:len(self.model_precomp.get_weights())])
        self.model_init.set_weights(weights[len(self.model_precomp.get_weights()):(len(self.model_precomp.get_weights()) + len(self.model_init.get_weights()))])
        self.model_prior.set_weights(weights[(len(self.model_precomp.get_weights()) + len(self.model_init.get_weights())):(len(self.model_precomp.get_weights()) + len(self.model_init.get_weights()) + len(self.model_prior.get_weights()))])
        self.model_data.set_weights(weights[(len(self.model_precomp.get_weights()) + len(self.model_init.get_weights()) + len(self.model_prior.get_weights())):])

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + self.shape

    def _merge_function(self, inputs):
        y, A = inputs
        uses_learning_phase = False

        # initialization
        batch_size = A.shape[0]

        i = K.constant(0, dtype=tf.int64)
        p = self.model_precomp_call(inputs)
        z_u = self.model_init_call(inputs + list(p))
        if (hasattr(self.model_precomp, 'activity_regularizer') and
           self.model_precomp.activity_regularizer is not None):
            for _p in p:
                regularization_loss = self.model_precomp.activity_regularizer(_p)
                self.add_loss(regularization_loss, inputs)
        if (hasattr(self.model_init, 'activity_regularizer') and
           self.model_init.activity_regularizer is not None):
            for _z_u in z_u:
                regularization_loss = self.model_init.activity_regularizer(_z_u)
                self.add_loss(regularization_loss, inputs)

        # unrolled iterations
        def body(i, z_u):
            global uses_learning_phase
            
            # x-update
            x = self.model_prior_call(list(z_u))
            if (hasattr(self.model_prior, 'activity_regularizer') and
                self.model_prior.activity_regularizer is not None):
                regularization_loss = self.model_prior.activity_regularizer(x)
                self.add_loss(regularization_loss, inputs)
            
            # z_u-update
            z_u = self.model_data_call(inputs + list(p) + [x] + list(z_u))
            if (hasattr(self.model_data, 'activity_regularizer') and
                self.model_data.activity_regularizer is not None):
                for _z_u in z_u:
                    regularization_loss = self.model_data.activity_regularizer(_z_u)
                    self.add_loss(regularization_loss, inputs)
            
            i = i + 1

            return i, z_u

        if self.unroll is False:
            cond = lambda i, z_u: tf.less(i, self.max_iter)
            shape_invar = tf.TensorShape([None, None]).concatenate(z_u[0].get_shape()[2:])
            shape_invariants = [i.get_shape(), shape_invar, shape_invar]
            i, z_u = tf.while_loop(cond, body, [i] + list(z_u), shape_invariants)
        else:
            for _ in range(self.max_iter):
                i, z_u = body(i, z_u)
        
        if uses_learning_phase:
            z_u._uses_learning_phase = True
            for _z_u in z_u:
                _z_u._uses_learning_phase = True
        return z_u[0]

    def call(self, inputs, mask=None, training=None, initial_state=None, constants=None):
        kwargs = {}
        #if has_arg(self.model_precomp.call, 'mask'):
        #    kwargs['mask'] = mask
        if has_arg(self.model_precomp.call, 'training'):
            kwargs['training'] = training
        if has_arg(self.model_precomp.call, 'constants'):
            kwargs['constants'] = constants
        self.model_precomp_call = lambda a: self.model_precomp.call(a, **kwargs)

        kwargs = {}
        #if has_arg(self.model_init.call, 'mask'):
        #    kwargs['mask'] = mask
        if has_arg(self.model_init.call, 'training'):
            kwargs['training'] = training
        if has_arg(self.model_init.call, 'constants'):
            kwargs['constants'] = constants
        self.model_init_call = lambda a: self.model_init.call(a, **kwargs)

        kwargs = {}
        #if has_arg(self.model_prior.call, 'mask'):
        #    kwargs['mask'] = mask
        if has_arg(self.model_prior.call, 'training'):
            kwargs['training'] = training
        if has_arg(self.model_prior.call, 'constants'):
            kwargs['constants'] = constants
        self.model_prior_call = lambda a: self.model_prior.call(a, **kwargs)

        kwargs = {}
        #if has_arg(self.model_data.call, 'mask'):
        #    kwargs['mask'] = mask
        if has_arg(self.model_data.call, 'training'):
            kwargs['training'] = training
        if has_arg(self.model_data.call, 'constants'):
            kwargs['constants'] = constants
        
        self.model_data_call = lambda a: self.model_data.call(a, **kwargs)
        return _Merge.call(self, inputs)

    def reset_states(self):
        self.model_precomp.reset_states()
        self.model_init.reset_states()
        self.model_prior.reset_states()
        self.model_data.reset_states()

    def get_config(self):
        config = {'model_precomp': {'class_name': self.model_precomp.__class__.__name__,
                                    'config': self.model_precomp.get_config()},
                  'model_init': {'class_name': self.model_init.__class__.__name__,
                                 'config': self.model_init.get_config()},
                  'model_prior': {'class_name': self.model_prior.__class__.__name__,
                                  'config': self.model_prior.get_config()},
                  'model_data': {'class_name': self.model_data.__class__.__name__,
                                 'config': self.model_data.get_config()},
                  'max_iter': self.max_iter,
                  'shape': self.shape,
                  'unroll': self.unroll}
        base_config = super(UnrolledOptimization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from . import deserialize as deserialize_layer
        model_precomp = deserialize_layer(config.pop('model_precomp'),
                                       custom_objects=custom_objects)
        model_init = deserialize_layer(config.pop('model_init'),
                                       custom_objects=custom_objects)
        model_prior = deserialize_layer(config.pop('model_prior'),
                                        custom_objects=custom_objects)
        model_data = deserialize_layer(config.pop('model_data'),
                                        custom_objects=custom_objects)
        return cls(model_precomp, model_init, model_prior, model_data, **config)


class InstanceNormalization(Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

