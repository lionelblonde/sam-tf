from mpi4py import MPI

import numpy as np
import tensorflow as tf

from imitation.helpers.tf_util import TheanoFunction


class MpiRunningMeanStd(object):

    def __init__(self, epsilon=1e-2, shape=()):
        """Relies on # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """

        self._sum = tf.get_variable(dtype=tf.float64,
                                    shape=shape,
                                    initializer=tf.constant_initializer(0.0),
                                    name="runningsum",
                                    trainable=False)
        self._sumsq = tf.get_variable(dtype=tf.float64,
                                      shape=shape,
                                      initializer=tf.constant_initializer(epsilon),
                                      name="runningsumsq",
                                      trainable=False)
        self._count = tf.get_variable(dtype=tf.float64,
                                      shape=(),
                                      initializer=tf.constant_initializer(epsilon),
                                      name="count", trainable=False)
        self.shape = shape

        self.mean = tf.to_float(self._sum / self._count)
        self.std = tf.sqrt(tf.maximum(tf.to_float(self._sumsq / self._count) -
                                      tf.square(self.mean),
                                      1e-2))

        self.newsum = tf.placeholder(shape=self.shape, dtype=tf.float64, name='sum')
        self.newsumsq = tf.placeholder(shape=self.shape, dtype=tf.float64, name='var')
        self.newcount = tf.placeholder(shape=[], dtype=tf.float64, name='count')

        updates = [tf.assign_add(self._sum, self.newsum),
                   tf.assign_add(self._sumsq, self.newsumsq),
                   tf.assign_add(self._count, self.newcount)]

        self.incfiltparams = TheanoFunction(inputs=[self.newsum, self.newsumsq, self.newcount],
                                            outputs=updates)

    def update(self, x, comm):
        x = x.astype('float64')
        n = int(np.prod(self.shape))
        totalvec = np.zeros(n * 2 + 1, 'float64')
        addvec = np.concatenate([x.sum(axis=0).ravel(),
                                 np.square(x).sum(axis=0).ravel(),
                                 np.array([len(x)], dtype='float64')])

        comm.Allreduce(addvec, totalvec, op=MPI.SUM)

        self.incfiltparams({self.newsum: totalvec[0:n].reshape(self.shape),
                            self.newsumsq: totalvec[n:2 * n].reshape(self.shape),
                            self.newcount: totalvec[2 * n]})
