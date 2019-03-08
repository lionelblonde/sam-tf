import numpy as np
import tensorflow as tf

from imitation.helpers.misc_util import numel
from imitation.helpers.mpi_moments import mpi_mean_like


class MpiAdamOptimizer(tf.train.AdamOptimizer):

    def __init__(self, comm, clip_norm=None, **kwargs):
        """Entension of the ADAM optimizer that performs parallel SGD
        consisting in averaging the gradients across mpi processes.
        """
        self.comm = comm
        self.clip_norm = clip_norm
        tf.train.AdamOptimizer.__init__(self, **kwargs)

    def compute_gradients(self, loss, var_list, **kwargs):
        """Override the ADAM optimizer standard function"""
        _grads_and_vars = tf.train.AdamOptimizer.compute_gradients(self, loss, var_list, **kwargs)
        grads_and_vars = [(grad_, var_) if grad_ is not None else (tf.zeros_like(var_), var_)
                          for (grad_, var_) in _grads_and_vars]
        flat_grad = tf.concat([tf.reshape(grad_, [numel(var_)])
                               for (grad_, var_) in grads_and_vars], axis=0)
        numels = [numel(var_) for _, var_ in grads_and_vars]
        # Wraps a python function and uses it as a TensorFlow op
        mean_flat_grad = tf.py_func(lambda x: mpi_mean_like(x, self.comm),
                                    [flat_grad], Tout=tf.float32)
        mean_flat_grad.set_shape(flat_grad.shape)
        mean_grads = tf.split(mean_flat_grad, numels, axis=0)
        if self.clip_norm is not None:
            # Clip the gradients by the ratio of the sum of their norms
            mean_grads, _ = tf.clip_by_global_norm(mean_grads, clip_norm=self.clip_norm)
        mean_grads_and_vars = [(tf.reshape(mean_grad_, var_.shape), var_)
                               for mean_grad_, (_, var_) in zip(mean_grads, grads_and_vars)]
        return mean_grads_and_vars

    def sync_from_root(self, var_list):
        """Send the root node parameters to every mpi worker"""
        self.comm.Barrier()
        rank = self.comm.Get_rank()
        # Reminder: for a Tensor t, calling t.eval() is equivalent to calling
        # tf.get_default_session().run(t)
        for var_ in var_list:
            if rank == 0:
                # Run the graph to get the value of the variable
                _var = var_.eval()
                # Broadcast the params from rank 0
                self.comm.Bcast(_var, root=0)
            else:
                _var_pulled = np.empty(var_.shape, dtype=np.float32)
                self.comm.Bcast(_var_pulled, root=0)
                tf.get_default_session().run(tf.assign(var_, _var_pulled))

    def check_synced(self, var_list):
        """Assert whether the workers' params have not strayed"""
        self.comm.Barrier()
        rank = self.comm.Get_rank()
        # Reminder: for a Tensor t, calling t.eval() is equivalent to calling
        # tf.get_default_session().run(t)
        if rank == 0:
            # Evaluate each var in the list and flatten into a 1-dim vector
            _var = np.concatenate([v.eval().ravel() for v in var_list], axis=0)
            # Broadcast the params from rank 0
            self.comm.Bcast(_var, root=0)
        else:
            # Evaluate each var in the list and flatten into a 1-dim vector
            _var_local = np.concatenate([v.eval().ravel() for v in var_list], axis=0)
            _var_root = np.empty_like(_var_local)
            self.comm.Bcast(_var_root, root=0)
            assert (_var_root == _var_local).all(), "mismatch {}\n{}".format(_var_root,
                                                                             _var_local)
