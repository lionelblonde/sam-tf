from mpi4py import MPI

import numpy as np


def mpi_mean_like(x, comm):
    """Computes element-wise mean across workers.
    The output array has the same shape as the input array.
    This operation will fail if the array can be of different shapes across the workers.
    e.g. used for gradient averaging across workers or when averagin scalars
    """
    assert comm is not None
    num_workers = comm.Get_size()
    x = np.asarray(x)
    # Initialize element-wise sums across the workers
    sums = np.empty_like(x)
    # Sum x's elements across all mpi workers and put the result in `sums`
    comm.Allreduce(x, sums, op=MPI.SUM)
    means = sums / num_workers
    # Make sure input and output have the same shape
    assert means.shape == x.shape
    return means


def mpi_mean_reduce_v(x, comm, axis=0, keepdims=False):
    """Compute mean locally along `axis` and globally across mpi workers.
    This is the verbose version (hence the 'v') as the number of reductions (local and
    global) is also returned in the output tuple.
    """
    assert comm is not None
    x = np.asarray(x)
    assert x.ndim >= 1
    # Collapse to x.ndim-1 dimensions by summin along `axis`
    sums = x.sum(axis=axis, keepdims=keepdims)
    # Extract the number of elements
    n = sums.size
    # Create a vector of size n+1, put flattened `sums` in the first `n` slots
    # and put how many elements were reduced along `axis` in the n+1-th slot
    # (i.e. the number of elements involved in each reduction)
    local_sums = np.zeros(n + 1, dtype=x.dtype)
    flattened_sums = sums.ravel()
    reduction_depth = x.shape[axis]
    local_sums[:n] = flattened_sums
    local_sums[n] = reduction_depth
    # Sum local_sums's elements across all mpi workers and put the result in `global_sum`
    global_sums = np.zeros_like(local_sums)
    comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
    # Unflatten the result (back to post-reduction along `axis` shape) and
    # divide by the sum across workers of numbers of reductions.
    # In fine, in the returned tensor, each element corresponds to a sum along axis (local)
    # and across workers (global) divided by the sum (across workers) of number of local
    # reductions.
    return global_sums[:n].reshape(sums.shape) / global_sums[n], global_sums[n]


def mpi_mean_reduce(x, comm, axis=0, keepdims=False):
    """Almost like 'mpi_mean_reduce_v', but only returns the mpi mean"""
    mpi_mean, _ = mpi_mean_reduce_v(x=x, comm=comm, axis=axis, keepdims=keepdims)
    return mpi_mean


def mpi_moments(x, comm, axis=0, keepdims=False):
    """Compute mpi moments"""
    assert comm is not None
    x = np.asarray(x)
    assert x.ndim >= 1
    # Compute mean
    mean, count = mpi_mean_reduce_v(x, axis=axis, comm=comm, keepdims=True)
    # Compute standard deviation
    squared_diffs = np.square(x - mean)
    mean_squared_diff, count1 = mpi_mean_reduce_v(squared_diffs, axis=axis,
                                                  comm=comm, keepdims=True)
    assert count1 == count1  # verify that nothing ominous happened when squaring
    std = np.sqrt(mean_squared_diff)
    if not keepdims:
        new_shape = mean.shape[:axis] + mean.shape[axis + 1:]
        mean = mean.reshape(new_shape)
        std = std.reshape(new_shape)
    return mean, std, count
