import copy
import os
import multiprocessing

import numpy as np
import tensorflow as tf

from imitation.helpers.misc_util import zipsame, numel, var_shape, intprod


class TheanoFunction(object):

        def __init__(self, inputs, outputs):
            """Drop in replacement for theano's `function`, the interface for
            compiling graphs into callable objects, in TensorFlow
            """
            for input_ in inputs:
                fmtbool = type(input_) is tf.Tensor and len(input_.op.inputs) == 0
                fmtstr = "inputs should all be phs, constants, or have a make_feed_dict method"
                assert hasattr(input_, 'make_feed_dict') or fmtbool, fmtstr
            self.inputs = inputs
            self.outputs = outputs

        def __call__(self, feed_dict):
            assert isinstance(feed_dict, dict), "can only feed with dictionary types"
            assert len(feed_dict) >= len(self.inputs), "need to feed every tensor"
            assert len(feed_dict) <= len(self.inputs), "too many values are being fed"
            return tf.get_default_session().run(self.outputs, feed_dict=feed_dict)


def switch(condition, then_expression, else_expression):
    """Switch between two operations depending on a scalar value (int or bool).
    Note that both `then_expression` and `else_expression` should be symbolic
    tensors of the *same shape*.

    # Arguments
        condition: scalar tensor,
        then_expression: TensorFlow operation,
        else_expression: TensorFlow operation.
    """
    x_shape = copy.copy(then_expression.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),
                lambda: then_expression,
                lambda: else_expression)
    x.set_shape(x_shape)

    return x


def clip(clip_range):
    """Clip the variable values to remain within the clip range `(l_b, u_b)`
    `l_b` and `u_b` respectively depict the lower and upper bounds of the range,
    e.g.
        clip_obs = clip(observation_range)
        clipped_obs = clip_obs(obs)
    """
    def _clip(x):
        return tf.clip_by_value(x, min(clip_range), max(clip_range))
    return _clip


def make_session(tf_config=None, num_core=None, make_default=False, graph=None):
    """Returns a session which will use <num_threads> threads only
    It does not always ensure better performance: https://stackoverflow.com/a/39395608
    Prefer MPI parallelism over tf
    """
    assert isinstance(tf_config, tf.ConfigProto) or tf_config is None
    if num_core is None:
        # Num of cores can also be communicated from the exterior, via an env var
        num_core = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    if tf_config is None:
        tf_config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=num_core,
            intra_op_parallelism_threads=num_core)
        tf_config.gpu_options.allow_growth = True
    if make_default:
        # The only difference with a regular Session is that
        # an InteractiveSession installs itself as the default session on construction
        return tf.InteractiveSession(config=tf_config, graph=graph)
    else:
        return tf.Session(config=tf_config, graph=graph)


def get_available_gpus():
    """Return the current available GPUs via tensorflow API
    From stackoverflow post:
        https://stackoverflow.com/questions/
        38559755/how-to-get-current-available-gpus-in-tensorflow?
        utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def single_threaded_session():
    """Returns a session which will only use a single core"""
    return make_session(num_core=1)


ALREADY_INITIALIZED = set()  # python set: unordered collection unique elements


def initialize():
    """Initialize all the uninitialized variables in the global scope"""
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    tf.get_default_session().run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)


def load_model(model, var_list=None):
    """Restore variables from disk
    `model` is the path of the model checkpoint to load
    """
    saver = tf.train.Saver(var_list=var_list)
    saver.restore(tf.get_default_session(), model)


def load_latest_checkpoint(model_dir, var_list=None):
    """Restore variables from disk
    `model_dir` is the path of the directory containing all the checkpoints
    for a given experiment
    """
    saver = tf.train.Saver(var_list=var_list)
    saver.restore(tf.get_default_session(), tf.train.latest_checkpoint(model_dir))


def save_state(fname, var_list=None, iters_so_far=None):
    """Save the variables to disk"""
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    # No exception is raised if the directory already exists
    saver = tf.train.Saver(var_list=var_list)
    saver.save(tf.get_default_session(), fname, global_step=iters_so_far)


def flatgrad(loss, var_list, clip_norm=None):
    """Returns a list of sum(dy/dx) for each x in `var_list`
    Clipping is done by global norm (paper: https://arxiv.org/abs/1211.5063)
    """
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
    vars_and_grads = zipsame(var_list, grads)  # zip with extra security
    for index, (var, grad) in enumerate(vars_and_grads):
        # If the gradient gets stopped for some obsure reason, set the grad as zero vector
        _grad = grad if grad is not None else tf.zeros_like(var)
        # Reshape the grad into a vector
        grads[index] = tf.reshape(_grad, [numel(var)])
    # return tf.concat(grads, axis=0)
    return tf.concat(grads, axis=0)


class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32):
        assigns = []
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])

        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = intprod(shape)
            assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)  # creates an op that groups multiple operations

    def __call__(self, theta):
        tf.get_default_session().run(self.op, feed_dict={self.theta: theta})


class GetFlat(object):
    def __init__(self, var_list):
        self.op = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])

    def __call__(self):
        return tf.get_default_session().run(self.op)


_PLACEHOLDER_CACHE = {}  # name -> (placeholder, dtype, shape)


def get_placeholder(name, dtype, shape):
    """Return a placeholder if already available in the current graph
    with the right shape. Otherwise, create the placeholder with the
    desired shape and return it.
    """
    if name in _PLACEHOLDER_CACHE:
        placeholder_, dtype_, shape_ = _PLACEHOLDER_CACHE[name]
        if placeholder_.graph == tf.get_default_graph():
            assert dtype_ == dtype and shape_ == shape, \
                "Placeholder with name {} has already been registered and has shape {}, \
                 different from requested {}".format(name, shape_, shape)
            return placeholder_
    placeholder_ = tf.placeholder(dtype=dtype, shape=shape, name=name)
    _PLACEHOLDER_CACHE[name] = (placeholder_, dtype, shape)
    return placeholder_


def get_placeholder_cached(name):
    """Returns an error if the placeholder does not exist"""
    return _PLACEHOLDER_CACHE[name][0]
