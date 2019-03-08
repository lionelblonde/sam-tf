import numpy as np
import tensorflow as tf


def layer_norm(inputs, axis=-1, offset=True, scale=True, name=None):
    """Layer Normalization
    Paper: https://arxiv.org/abs/1607.06450

    Transforms input x into: outputs = gamma * (x - mu) / sigma + beta
    where mu and sigma are respectively the mean and standard deviation of x.
    Gamma and beta are trainable parameters for scaling and shifting respectively.

    Since the axes over which normalization is perfomed is configurable, this also
    subsumes instance normalization.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # Define the beta and gamma parameters of the normalization
        beta, gamma = None, None
        if offset:
            beta = tf.get_variable(name='beta',
                                   shape=inputs.shape[axis],
                                   dtype=tf.float32,
                                   initializer=tf.zeros_initializer(),
                                   trainable=True)
        if scale:
            gamma = tf.get_variable(name='gamma',
                                    shape=inputs.shape[axis],
                                    dtype=tf.float32,
                                    initializer=tf.ones_initializer(),
                                    trainable=True)
        # Calculate the moments on the last axis (layer activations)
        mean, variance = tf.nn.moments(inputs, axes=[axis], keep_dims=True)
        # Compute layer normalization using the batch_normalization function
        outputs = tf.nn.batch_normalization(inputs,
                                            mean=mean,
                                            variance=variance,
                                            offset=beta,
                                            scale=gamma,
                                            variance_epsilon=1e-8)
    return outputs


def weight_decay_regularizer(scale):
    """Apply l2 regularization on weights"""
    assert scale >= 0
    if scale == 0:
        return lambda _: None

    def _regularizer(weights):
        scale_tensor = tf.convert_to_tensor(scale,
                                            dtype=weights.dtype.base_dtype,
                                            name='weight_decay_scale')
        wd_loss = tf.multiply(scale_tensor, tf.nn.l2_loss(weights), name='weight_decay_loss')
        return wd_loss

    return _regularizer


def leaky_relu(x, leak=0.2, name='leaky_relu'):
    """Leaky ReLU activation function
    'Rectifier Nonlinearities Improve Neural Network Acoustic Models'
    AL Maas, AY Hannun, AY Ng, ICML, 2013
    http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf

    Alternate implementation that might be more efficient:
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
    [relies on max(0, x) = (x + abs(x))/2]
    """
    with tf.variable_scope(name):
        return tf.maximum(x, leak * x)


def prelu(x, name='prelu'):
    """Parametric ReLU activation function
    'Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification'
    He Kaiming, ICCV 2015, http://arxiv.org/abs/1502.01852
    """
    with tf.variable_scope(name):
        leak = tf.get_variable('leak', x.get_shape()[-1], initializer=tf.zeros_initializer())
        return tf.maximum(x, leak * x)


def selu(x, name='selu'):
    """Scaled ELU activation function
    'Self-Normalizing Neural Networks'
    Günter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter, NIPS 2017,
    https://arxiv.org/abs/1706.02515
    Correct alpha and scale values for unit activation with mean zero and unit variance
    from https://github.com/bioinf-jku/SNNs/blob/master/selu.py#L24:9
    """
    with tf.variable_scope(name):
        # Values of alpha and scale corresponding to zero mean and unit variance
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        # The authors' implementation returns
        #     scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))
        # We replace tf.nn.elu by its expression
        #     tf.exp(x) - 1.     for x < 0
        #     x                  for x >= 0
        # By substitution, scale * tf.where(x >= 0., x, alpha * (tf.exp(x) - 1.))
        return scale * tf.where(x >= 0., x, alpha * tf.exp(x) - alpha)


def parse_nonlin(nonlin_key):
    """Parse the activation function"""
    nonlin_map = dict(relu=tf.nn.relu,
                      leaky_relu=leaky_relu,
                      prelu=prelu,
                      elu=tf.nn.elu,
                      selu=selu,
                      tanh=tf.nn.tanh,
                      identity=tf.identity)
    if nonlin_key in nonlin_map.keys():
        return nonlin_map[nonlin_key]
    else:
        raise RuntimeError("unknown nonlinearity: '{}'".format(nonlin_key))


def extract_fan_in_out(shape):
    fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
    fan_out = float(shape[-1])
    for dim in shape[:-2]:
        fan_in *= float(dim)
        fan_out *= float(dim)
    return fan_in, fan_out


def xavier_uniform_init():
    """Xavier uniform initialization
    'Understanding the difficulty of training deep feedforward neural networks'
    Xavier Glorot & Yoshua Bengio, AISTATS 2010,
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    Draws samples from a uniform distribution within [-w_bound, w_bound],
    with w_bound = np.sqrt(6.0 / (fan_in + fan_out))
    """
    def _initializer(shape, dtype=None, partition_info=None):
        fan_in, fan_out = extract_fan_in_out(shape)
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform(shape=shape, minval=-w_bound, maxval=w_bound, dtype=dtype)
    return _initializer


def xavier_normal_init():
    """Xavier normal initialization
    'Understanding the difficulty of training deep feedforward neural networks'
    Xavier Glorot & Yoshua Bengio, AISTATS 2010,
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    Draws samples from a truncated normal distribution centered on 0,
    with stddev = np.sqrt(2.0 / (fan_in + fan_out)) i.e. np.sqrt(1.0 / fan_avg)
    """
    def _initializer(shape, dtype=None, partition_info=None):
        fan_in, fan_out = extract_fan_in_out(shape)
        # Without truncation: stddev = np.sqrt(2.0 / (fan_in + fan_out))
        trunc_stddev = np.sqrt(1.3 * 2.0 / (fan_in + fan_out))
        return tf.truncated_normal(shape=shape, mean=0.0, stddev=trunc_stddev, dtype=dtype)
    return _initializer


def he_uniform_init():
    """He (MSRA) uniform initialization
    'Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification'
    Kaiming He, ICCV 2015, http://arxiv.org/abs/1502.01852
    Xavier initialization is justified when only linear activations are used,
    He initialization is justified when ReLU/leaky RELU/PReLU activations are used.

    Draws samples from a uniform distribution within [-w_bound, w_bound],
    with w_bound = np.sqrt(6.0 / fan_in)
    """
    def _initializer(shape, dtype=None, partition_info=None):
        fan_in, _ = extract_fan_in_out(shape)
        w_bound = np.sqrt(6.0 / fan_in)
        return tf.random_uniform(shape=shape, minval=-w_bound, maxval=w_bound, dtype=dtype)
    return _initializer


def he_normal_init():
    """He (MSRA) normal initialization
    'Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification'
    He Kaiming, ICCV 2015, http://arxiv.org/abs/1502.01852
    Xavier initialization is justified when only linear activations are used,
    He initialization is justified when ReLU/leaky RELU/PReLU activations are used.

    Draws samples from a truncated normal distribution centered on 0,
    with stddev = np.sqrt(2.0 / fan_in)
    """
    def _initializer(shape, dtype=None, partition_info=None):
        fan_in, _ = extract_fan_in_out(shape)
        # Without truncation: stddev = np.sqrt(2.0 / fan_in)
        trunc_stddev = np.sqrt(1.3 * 2.0 / fan_in)
        return tf.truncated_normal(shape=shape, mean=0.0, stddev=trunc_stddev, dtype=dtype)
    return _initializer


def parse_initializer(hid_w_init_key):
    """Parse the weight initializer"""
    init_map = dict(he_normal=he_normal_init(),
                    he_uniform=he_uniform_init(),
                    xavier_normal=xavier_normal_init(),
                    xavier_uniform=xavier_uniform_init())
    if hid_w_init_key in init_map.keys():
        return init_map[hid_w_init_key]
    else:
        raise RuntimeError("unknown weight init: '{}'".format(hid_w_init_key))
