import tensorflow as tf

from gym import spaces

from imitation.helpers.network_util import (layer_norm, weight_decay_regularizer,
                                            parse_nonlin, parse_initializer)
from imitation.helpers.misc_util import zipsame


class AbstractNN(object):

    def __init__(self):
        pass

    def stack_conv2d_layers(self, embedding):
        """Stack the Conv2D layers"""
        conv2dz = zipsame(self.hps.nums_filters, self.hps.filter_shapes, self.hps.stride_shapes)
        for conv2d_layer_index, zipped_conv2d in enumerate(conv2dz, start=1):
            conv2d_layer_id = "conv2d{}".format(conv2d_layer_index)
            num_filters, filter_shape, stride_shape = zipped_conv2d  # unpack
            # Add cond2d hidden layer and non-linearity
            embedding = tf.layers.conv2d(inputs=embedding,
                                         filters=num_filters,
                                         kernel_size=filter_shape,
                                         strides=stride_shape,
                                         padding='valid',
                                         data_format='channels_last',
                                         dilation_rate=(1, 1),
                                         activation=parse_nonlin(self.hps.hid_nonlin),
                                         use_bias=True,
                                         kernel_initializer=self.hid_initializers['w'],
                                         bias_initializer=self.hid_initializers['b'],
                                         kernel_regularizer=self.hid_regularizers['w'],
                                         bias_regularizer=None,
                                         activity_regularizer=None,
                                         kernel_constraint=None,
                                         bias_constraint=None,
                                         trainable=True,
                                         name=conv2d_layer_id,
                                         reuse=tf.AUTO_REUSE)
        # Flatten between conv2d layers and fully-connected layers
        embedding = tf.layers.flatten(inputs=embedding, name='flatten')
        return embedding

    def stack_fc_layers(self, embedding):
        """Stack the fully-connected layers
        Note that according to the paper 'Parameter Space Noise for Exploration', layer
        normalization should only be used for the fully-connected part of the network.
        """
        for hid_layer_index, hid_width in enumerate(self.hps.hid_widths, start=1):
            hid_layer_id = "fc{}".format(hid_layer_index)
            # Add hidden layer and non-linearity
            embedding = tf.layers.dense(inputs=embedding,
                                        units=hid_width,
                                        activation=parse_nonlin(self.hps.hid_nonlin),
                                        use_bias=True,
                                        kernel_initializer=self.hid_initializers['w'],
                                        bias_initializer=self.hid_initializers['b'],
                                        kernel_regularizer=self.hid_regularizers['w'],
                                        bias_regularizer=None,
                                        activity_regularizer=None,
                                        kernel_constraint=None,
                                        bias_constraint=None,
                                        trainable=True,
                                        name=hid_layer_id,
                                        reuse=tf.AUTO_REUSE)
            if self.hps.with_layernorm:
                # Add layer normalization
                ln_layer_id = "layernorm{}".format(hid_layer_index)
                embedding = layer_norm(embedding, name=ln_layer_id)

        return embedding

    def set_hid_initializers(self):
        self.hid_initializers = {'w': parse_initializer(self.hps.hid_w_init),
                                 'b': tf.zeros_initializer()}

    def set_out_initializers(self):
        self.out_initializers = {'w': tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                                 'b': tf.zeros_initializer()}

    def set_hid_regularizer(self):
        self.hid_regularizers = {'w': weight_decay_regularizer(scale=0.)}

    def add_out_layer(self, embedding):
        pass

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=self.scope + "/" + self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 scope=self.scope + "/" + self.name)


class PolicyNN(AbstractNN):

    def __init__(self, scope, name, ac_space, hps):
        self.scope = scope
        self.name = name
        self.ac_space = ac_space
        self.hps = hps
        self.set_pd_type()

    def __call__(self, ob):
        with tf.variable_scope(self.name):
            self.set_hid_initializers()
            self.set_out_initializers()
            self.set_hid_regularizer()
            embedding = ob
            if self.hps.from_raw_pixels:
                embedding = self.stack_conv2d_layers(embedding)
            embedding = self.stack_fc_layers(embedding)
            pd_params = self.add_out_layer(embedding)
            # Return the probability distribution over actions
            pd = self.pd_type.pdfromflat(pd_params)
        return pd

    def set_pd_type(self):
        """Create `pd_type` based on the action space.
        Covers gaussian policies for continuous action spaces (e.g. MuJoCo)
        and categorical policies for discrete action spaces (e.g. ALE)"""
        from imitation.helpers.distributions import DiagGaussianPdType, CategoricalPdType
        if isinstance(self.ac_space, spaces.Box):
            self.ac_dim = self.ac_space.shape[-1]  # num dims
            self.pd_type = DiagGaussianPdType(self.ac_dim)
        elif isinstance(self.ac_space, spaces.Discrete):
            self.ac_dim = self.ac_space.n  # num ac choices
            self.pd_type = CategoricalPdType(self.ac_dim)
        else:
            raise RuntimeError("ac space is neither Box nor Discrete")

    def set_out_initializers(self):
        """Set output initializers"""
        self.out_initializers = {'w': tf.truncated_normal_initializer(stddev=0.01),
                                 'b': tf.zeros_initializer()}

    def add_out_layer(self, embedding):
        """Add the output layer"""
        if isinstance(self.ac_space, spaces.Box) and self.hps.gaussian_fixed_var:
            half_pd_param_len = self.pd_type.param_shape()[0] // 2
            mean = tf.layers.dense(inputs=embedding,
                                   units=half_pd_param_len,
                                   activation=None,
                                   use_bias=True,
                                   kernel_initializer=self.out_initializers['w'],
                                   bias_initializer=self.out_initializers['b'],
                                   kernel_regularizer=None,
                                   bias_regularizer=None,
                                   activity_regularizer=None,
                                   kernel_constraint=None,
                                   bias_constraint=None,
                                   trainable=True,
                                   name='final',
                                   reuse=tf.AUTO_REUSE)
            log_std = tf.get_variable(shape=[1, half_pd_param_len], name='log_std',
                                      initializer=tf.zeros_initializer())
            # Concat mean and std
            # "What is what?"" in the params is done in the pd sampling function, not here
            pd_params = tf.concat([mean, mean * 0.0 + log_std], axis=1)
            # HAXX: w/o `mean * 0.0 +` tf does not accept as mean is [None, ...] >< [1, ...]
            # -> broadcasting haxx
        else:
            pd_params = tf.layers.dense(inputs=embedding,
                                        units=self.pd_type.param_shape()[0],
                                        activation=None,
                                        use_bias=True,
                                        kernel_initializer=self.out_initializers['w'],
                                        bias_initializer=self.out_initializers['b'],
                                        kernel_regularizer=None,
                                        bias_regularizer=None,
                                        activity_regularizer=None,
                                        kernel_constraint=None,
                                        bias_constraint=None,
                                        trainable=True,
                                        name='final',
                                        reuse=tf.AUTO_REUSE)
        return pd_params


class ValueNN(AbstractNN):

    def __init__(self, scope, name, hps):
        self.scope = scope
        self.name = name
        self.hps = hps

    def __call__(self, ob):
        with tf.variable_scope(self.name):
            self.set_hid_initializers()
            self.set_out_initializers()
            self.set_hid_regularizer()
            embedding = ob
            if self.hps.from_raw_pixels:
                embedding = self.stack_conv2d_layers(embedding)
            embedding = self.stack_fc_layers(embedding)
            v = self.add_out_layer(embedding)
        return v

    def add_out_layer(self, embedding):
        """Add the output layer"""
        embedding = tf.layers.dense(inputs=embedding,
                                    units=1,
                                    activation=None,
                                    use_bias=True,
                                    kernel_initializer=self.out_initializers['w'],
                                    bias_initializer=self.out_initializers['b'],
                                    kernel_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    kernel_constraint=None,
                                    bias_constraint=None,
                                    trainable=True,
                                    name='final',
                                    reuse=tf.AUTO_REUSE)
        return embedding


class RewardNN(AbstractNN):

    def __init__(self, scope, name, hps):
        self.scope = scope
        self.name = name
        self.hps = hps

    def __call__(self, obs, acs):
        with tf.variable_scope(self.name):
            # Concatenate the observations and actions placeholders to form a pair
            self.set_hid_initializers()
            self.set_out_initializers()
            self.set_hid_regularizer()
            embedding = obs
            if self.hps.from_raw_pixels:
                embedding = self.stack_conv2d_layers(embedding)
            embedding = tf.concat([embedding, acs], axis=-1)  # preserves batch size
            embedding = self.stack_fc_layers(embedding)
            scores = self.add_out_layer(embedding)
        return scores

    def add_out_layer(self, embedding):
        """Add the output layer"""
        embedding = tf.layers.dense(inputs=embedding,
                                    units=1,
                                    activation=None,
                                    use_bias=True,
                                    kernel_initializer=self.out_initializers['w'],
                                    bias_initializer=self.out_initializers['b'],
                                    kernel_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    kernel_constraint=None,
                                    bias_constraint=None,
                                    trainable=True,
                                    name='final',
                                    reuse=tf.AUTO_REUSE)
        return embedding


class ActorNN(AbstractNN):

    def __init__(self, scope, name, ac_space, hps):
        self.scope = scope
        self.name = name
        self.ac_space = ac_space
        self.hps = hps

    def __call__(self, ob):
        with tf.variable_scope(self.name):
            self.set_hid_initializers()
            self.set_out_initializers()
            self.set_hid_regularizer()
            embedding = ob
            if self.hps.from_raw_pixels:
                embedding = self.stack_conv2d_layers(embedding)
            embedding = self.stack_fc_layers(embedding)
            ac = self.add_out_layer(embedding)
        return ac

    def add_out_layer(self, embedding):
        if isinstance(self.ac_space, spaces.Box):
            self.ac_dim = self.ac_space.shape[-1]  # num dims
            # Add the output layer
            # and apply tanh as output nonlinearity
            embedding = tf.layers.dense(inputs=embedding,
                                        units=self.ac_dim,
                                        activation=tf.nn.tanh,
                                        use_bias=True,
                                        kernel_initializer=self.out_initializers['w'],
                                        bias_initializer=self.out_initializers['b'],
                                        kernel_regularizer=None,
                                        bias_regularizer=None,
                                        activity_regularizer=None,
                                        kernel_constraint=None,
                                        bias_constraint=None,
                                        trainable=True,
                                        name='final',
                                        reuse=tf.AUTO_REUSE)
            # Scale with maximum ac value
            embedding = self.ac_space.high * embedding
        elif isinstance(self.ac_space, spaces.Discrete):
            self.ac_dim = self.ac_space.n  # num ac choices
            # Add the output layer
            # and apply softmax as output nonlinearity (prob of playing each discrete action)
            embedding = tf.layers.dense(inputs=embedding,
                                        units=self.ac_dim,
                                        activation=lambda x: tf.nn.softmax(x, axis=-1),
                                        use_bias=True,
                                        kernel_initializer=self.out_initializers['w'],
                                        bias_initializer=self.out_initializers['b'],
                                        kernel_regularizer=None,
                                        bias_regularizer=None,
                                        activity_regularizer=None,
                                        kernel_constraint=None,
                                        bias_constraint=None,
                                        trainable=True,
                                        name='final',
                                        reuse=tf.AUTO_REUSE)
        else:
            raise RuntimeError("ac space is neither Box nor Discrete")
        return embedding

    @property
    def perturbable_vars(self):
        """Following the paper 'Parameter Space Noise for Exploration', we do not
        perturb the conv2d layers, only the fully-connected part of the network.
        Additionally, the extra variables introduced by layer normalization should remain
        unperturbed as they do not play any role in exploration. The only variables that
        we want to perturb are the weights and biases of the fully-connected layers.
        """
        return [var for var in self.trainable_vars if ('layernorm' not in var.name and
                                                       'conv2d' not in var.name)]


class CriticNN(AbstractNN):

    def __init__(self, scope, name, hps):
        self.scope = scope
        self.name = name
        self.hps = hps
        assert self.hps.ac_branch_in <= len(self.hps.hid_widths)

    def __call__(self, ob, ac):
        with tf.variable_scope(self.name):
            self.set_hid_initializers()
            self.set_out_initializers()
            self.set_hid_regularizer()
            embedding = ob
            if self.hps.from_raw_pixels:
                embedding = self.stack_conv2d_layers(embedding)
            embedding = self.stack_fc_layers(embedding, ac)
            q = self.add_out_layer(embedding)
        return q

    def stack_fc_layers(self, embedding, ac):
        """Stack the fully-connected layers"""
        for hid_layer_index, hid_width in enumerate(self.hps.hid_widths, start=1):
            hid_layer_id = "fc{}".format(hid_layer_index)
            if hid_layer_index == self.hps.ac_branch_in:
                # Concat ac to features extracted from ob
                embedding = tf.concat([embedding, ac], axis=-1)  # preserves batch size
            # Add hidden layer and apply non-linearity
            embedding = tf.layers.dense(inputs=embedding,
                                        units=hid_width,
                                        activation=parse_nonlin(self.hps.hid_nonlin),
                                        use_bias=True,
                                        kernel_initializer=self.out_initializers['w'],
                                        bias_initializer=self.out_initializers['b'],
                                        kernel_regularizer=self.hid_regularizers['w'],
                                        bias_regularizer=None,
                                        activity_regularizer=None,
                                        kernel_constraint=None,
                                        bias_constraint=None,
                                        trainable=True,
                                        name=hid_layer_id,
                                        reuse=tf.AUTO_REUSE)
            if self.hps.with_layernorm:
                # Add layer normalization
                ln_layer_id = "layernorm{}".format(hid_layer_index)
                embedding = layer_norm(embedding, name=ln_layer_id)
        return embedding

    def set_hid_regularizer(self):
        self.hid_regularizers = {'w': weight_decay_regularizer(scale=self.hps.wd_scale)}

    def add_out_layer(self, embedding):
        """Add the output layer"""
        embedding = tf.layers.dense(inputs=embedding,
                                    units=1,
                                    activation=None,
                                    use_bias=True,
                                    kernel_initializer=self.out_initializers['w'],
                                    bias_initializer=self.out_initializers['b'],
                                    kernel_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    kernel_constraint=None,
                                    bias_constraint=None,
                                    trainable=True,
                                    name='final',
                                    reuse=tf.AUTO_REUSE)
        return embedding

    @property
    def decayable_vars(self):
        """Variables on which weight decay can be applied"""
        return [var for var in self.trainable_vars if ('kernel' in var.name and
                                                       'layernorm' not in var.name and
                                                       'conv2d' not in var.name and
                                                       'final' not in var.name)]

    @property
    def regularization_losses(self):
        return tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 scope=self.scope + "/" + self.name)

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'final' in var.name]
        return output_vars
