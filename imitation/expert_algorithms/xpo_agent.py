import numpy as np
import tensorflow as tf

from gym import spaces

from imitation.helpers.tf_util import clip, get_placeholder, switch, TheanoFunction
from imitation.helpers.console_util import log_module_info
from imitation.helpers.math_util import rmsify
from imitation.helpers.networks import PolicyNN, ValueNN
from imitation.helpers.mpi_running_mean_std import MpiRunningMeanStd
from imitation.helpers import logger


class XPOAgent(object):

    def __init__(self, name, env, hps):
        self.name = name
        # Define everything in a specific scope
        with tf.variable_scope(self.name):
            self.scope = tf.get_variable_scope().name
            self._init(env=env, hps=hps)

    def _init(self, env, hps):
        self.env = env
        self.ob_shape = self.env.observation_space.shape
        self.ac_space = self.env.action_space
        self.hps = hps

        # Assemble clipping functions
        if isinstance(self.ac_space, spaces.Box):
            self.clip_obs = clip((-5., 5.))
        elif isinstance(self.ac_space, spaces.Discrete):
            self.clip_obs = clip((-np.infty, np.infty))
        else:
            raise RuntimeError("ac space is neither Box nor Discrete")

        # Define the policy and value networks
        self.policy_nn = PolicyNN(scope=self.scope, name='pol', ac_space=self.ac_space, hps=hps)
        self.value_nn = ValueNN(scope=self.scope, name='vf', hps=hps)

        self.pd_type = self.policy_nn.pd_type  # used outside for ac ph init

        # Create inputs
        self.ob = get_placeholder(name='ob', dtype=tf.float32, shape=(None,) + self.ob_shape)
        self.sample_or_mode = get_placeholder(name="sample_or_mode", dtype=tf.bool, shape=())

        # Rescale observations
        if self.hps.from_raw_pixels:
            # Scale pixel values
            obz = tf.cast(self.ob, tf.float32) / 255.0
        else:
            if self.hps.rmsify_obs:
                # Smooth out observations using running statistics and clip
                with tf.variable_scope("apply_obs_rms"):
                    self.obs_rms = MpiRunningMeanStd(shape=self.ob_shape)
                obz = self.clip_obs(rmsify(self.ob, self.obs_rms))
            else:
                obz = self.ob

        # Build graph
        self.pd_pred = self.policy_nn(obz)
        self.ac_pred = switch(self.sample_or_mode, self.pd_pred.sample(), self.pd_pred.mode())
        self.v_pred = self.value_nn(obz)

        # Create a callable object that predicts action and state value for the current state
        self.act_compute_v = TheanoFunction(inputs=[self.sample_or_mode, self.ob],
                                            outputs=[self.ac_pred, self.v_pred])

        # Summarize module information in logs
        log_module_info(logger, self.name, self.policy_nn, self.value_nn)

    def predict(self, sample_or_mode, ob):
        """Act and compute state value from a single observation
        The networks are able to process observations in minibatches, but the RL paradigm
        enforces the agent to see observations in sequences, therefore seeing only one
        at a time.
        `ob` is structured as np.array([a, b, c, ...]). Since the networks work with minibatches,
        we have to construct a minibatch of size 1, e.g. by using `ob[None]` (do not use `[ob]`!)
        which is structured as np.array([[a, b, c, ...]]).
        The networks both output an entity per observation in the input minibatch. Their
        respective outputs are:
            - `ac_pred`: a single action for the single provided observation:
                np.array([[d, e, f, ...]])
            - `v_pred`: a single state value for the single provided observation:
                np.array([[g]])
        Since a minibatch will later be sequentially construted out of the outputs, we extract
        the output from the returned minibatch of size 1. The extraction can be done by taking
        the first element (one or several times) with `.[0]` or by collapsing the returned array
        into one dimension with numpy's `flatten()` function:
            - `ac_pred`: np.array([[d, e, f, ...]]) -> np.array([d, e, f, ...])
            - `v_pred`: np.array([[g]]) -> np.array([g])
        Note that we only manipulate numpy arrays and not lists. We therefore do not need to
        extract the scalar `g` from `np.array([g])`, as for arithmetic operations a numpy array
        of size 1 is equivalent to a scalar (`np.array([1]) + np.array([2]) = np.array([3])` but
        `[1] + [2] = [1, 2]`).
        For safety reason, the scalar is still extracted from the singleton numpy array with
        `np.asscalar`.
        """
        ac_pred, v_pred = self.act_compute_v({self.sample_or_mode: sample_or_mode,
                                              self.ob: ob[None]})

        return ac_pred.flatten(), np.asscalar(v_pred.flatten())

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

    @property
    def pol_vars(self):
        return self.policy_nn.vars

    @property
    def pol_trainable_vars(self):
        return self.policy_nn.trainable_vars

    @property
    def vf_vars(self):
        return self.value_nn.vars

    @property
    def vf_trainable_vars(self):
        return self.value_nn.trainable_vars
