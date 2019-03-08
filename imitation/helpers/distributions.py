import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops


class Pd(object):
    """Probability distribution"""

    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def logp(self, x):
        return - self.neglogp(x)


class PdType(object):
    """Parametrized family of probability distributions"""

    def pdclass(self):
        raise NotImplementedError

    def pdfromflat(self, flat):
        return self.pdclass()(flat)

    def param_shape(self):
        raise NotImplementedError

    def sample_shape(self):
        raise NotImplementedError

    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape + self.param_shape(),
                              name=name)

    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape + self.sample_shape(),
                              name=name)


class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat

    def pdclass(self):
        return CategoricalPd

    def param_shape(self):
        return [self.ncat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return tf.int32


class MultiCategoricalPdType(PdType):
    def __init__(self, nvec):
        self.ncats = nvec

    def pdclass(self):
        return MultiCategoricalPd

    def pdfromflat(self, flat):
        return MultiCategoricalPd(self.ncats, flat)

    def param_shape(self):
        return [sum(self.ncats)]

    def sample_shape(self):
        return [len(self.ncats)]

    def sample_dtype(self):
        return tf.int32


class DiagGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return DiagGaussianPd

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32


class BernoulliPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return BernoulliPd

    def param_shape(self):
        return [self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.int32


class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    def neglogp(self, x):
        """Return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        Note that we can't use sparse_softmax_cross_entropy_with_logits because the implementation
        does not allow second-order derivatives...
        """
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                          labels=one_hot_actions)

    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def sample(self):
        """Technically this function does not sample from a categorical,
        but uses the Gumbel-Max trick.
        'argmax' being a non-differentiable function, most gradients flowing back through
        this operation will be filled with zeros. This sampling function can however be
        use whenever the loss being differentiated does not involve the sample itself
        but the parameters of the distribution (e.g. ppo/trpo's loss uses the logprob,
        which is a function of the parameter, not the result of `sample`).
        """
        u = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

    def sample2(self, temperature=0.3):
        """Technically this function does not sample from a categorical,
        but uses the Gumbel-Softmax trick, soft sampling counterpart of the Gumbel-Max trick,
        and provides a differentiable approximation of the Gumbel-Max trick,
        by replacing the argmax with a softmax.
        We use the argmax in the forward pass and the softmax (C0 approx) in the backward pass.
        Source: http://blog.evjang.com/2016/11/tutorial-categorical-variational.html
        """
        self.temperature = temperature

        def sample_gumbel(logits, eps=1e-20):
            """Sample from Gumbel(0, 1)"""
            u = tf.random_uniform(tf.shape(logits))
            return -tf.log(-tf.log(u + eps) + eps)

        def gumbel_softmax_sample(logits, temperature):
            """ Draw a sample from the Gumbel-Softmax distribution"""
            y = logits + sample_gumbel(logits)
            return tf.nn.softmax(y / temperature)

        def gumbel_softmax(logits, temperature, hard=True):
            """Sample from the Gumbel-Softmax distribution and optionally discretize.

            Args:
                logits (Tensor): Unnormalized log-probs
                temperature (float): Temperature
                hard (bool): If True, take argmax, but differentiate w.r.t. soft sample y

            Returns:
                Tensor corresponding to a sample from the Gumbel-Softmax distribution.
                If hard=True, then the returned sample will be one-hot, otherwise it will
                be a probabilitiy distribution that sums to 1 across classes
            """
            y = gumbel_softmax_sample(logits, temperature)
            if hard:
                # Create a vector whose elements are 0 when equal to to the maximum
                # Hence, `y_hard` is a one-hot vector of same dimension as `y`, (and same type)
                y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, axis=1, keepdims=True)), y.dtype)
                # Hack to use the argmax in forward pass but softmax in backward pass
                y = tf.stop_gradient(y_hard - y) + y
            return y

        # Return a categorical sample, while making `sample` a differentiable op
        # While `tf.argmax` returns the index of the max element, the current implementation
        # returns a vector (one-hot in forward pass if hard=True)
        return gumbel_softmax(self.logits, self.temperature)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


class MultiCategoricalPd(Pd):
    def __init__(self, nvec, flat):
        self.flat = flat
        self.categoricals = list(map(CategoricalPd, tf.split(flat, nvec, axis=-1)))

    def flatparam(self):
        return self.flat

    def mode(self):
        return tf.cast(tf.stack([p.mode() for p in self.categoricals], axis=-1), dtype=tf.int32)

    def neglogp(self, x):
        return tf.add_n([p.neglogp(px) for p, px in zip(self.categoricals,
                                                        tf.unstack(x, axis=-1))])

    def kl(self, other):
        return tf.add_n([p.kl(q) for p, q in zip(self.categoricals, other.categoricals)])

    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])

    def sample(self):
        return tf.cast(tf.stack([p.sample() for p in self.categoricals], axis=-1), dtype=tf.int32)

    @classmethod
    def fromflat(cls, flat):
        raise NotImplementedError


class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.flat = flat  # flattened concatenated params into a param vector
        # Split the param vector in 2: first half is used as mean, second as std
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def flatparam(self):
        return self.flat

    def mode(self):
        return self.mean

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
            + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
            + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return tf.reduce_sum(other.logstd -
                             self.logstd +
                             (tf.square(self.std) + tf.square(self.mean - other.mean)) /
                             (2.0 * tf.square(other.std)) -
                             0.5,
                             axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        # Use the reparameterization trick
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


class BernoulliPd(Pd):
    def __init__(self, logits):
        self.logits = logits
        self.ps = tf.sigmoid(logits)

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.round(self.ps)

    def neglogp(self, x):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                     labels=tf.to_float(x)),
                             axis=-1)

    def kl(self, other):
        aux1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=other.logits,
                                                                     labels=self.ps),
                             axis=-1)
        aux2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                     labels=self.ps),
                             axis=-1)
        return aux1 - aux2

    def entropy(self):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                     labels=self.ps),
                             axis=-1)

    def sample(self):
        u = tf.random_uniform(tf.shape(self.ps))
        return tf.to_float(math_ops.less(u, self.ps))  # truth value of (x < y) element-wise

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


def make_pdtype(ac_space):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalPdType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalPdType(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliPdType(ac_space.n)
    else:
        raise NotImplementedError
