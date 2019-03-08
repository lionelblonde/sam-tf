import numpy as np
import tensorflow as tf
import scipy.signal

from imitation.helpers.console_util import columnize


def rmsify(x, x_rms):
    """Normalize `x` with running statistics"""
    assert x.dtype == tf.float32, "must be a tensor of the right dtype"
    rmsed_x = (x - x_rms.mean) / x_rms.std
    return rmsed_x


def dermsify(x, x_rms):
    """Denormalize `x` with running statistics"""
    assert x.dtype == tf.float32, "must be a tensor of the right dtype"
    dermsed_x = (x * x_rms.std) + x_rms.mean
    return dermsed_x


def huber_loss(x, delta=1.0, name='huber_loss'):
    """Less sensitive to outliers than the l2 loss while being differentiable at 0
    Reference: https://en.wikipedia.org/wiki/Huber_loss
    """
    with tf.variable_scope(name):
        return tf.where(tf.abs(x) < delta,
                        0.5 * tf.square(x),
                        delta * (tf.abs(x) - 0.5 * delta))


def discount(x, gamma):
    """Compute discounted sum along the 0-th dimension of the `x` ndarray
    Return an ndarray `y` with the same shape as x, satisfying:
        y[t] = x[t] + gamma * x[t+1] + gamma^2 * x[t+2] + ... + gamma^k * x[t+k],
            where k = len(x) - t - 1

    Args:
        x (np.ndarray): 2-D array of floats, time x features
        gamma (float): Discount factor
    """
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def augment_segment_gae_stats(segment, gamma, lam, rew_key):
    """'Generalized Advantage Estimation' (GAE)
    Schulman, ICLR 2016, https://arxiv.org/abs/1506.02438

    Args:
        segment (dict): Collected segment of transitions (>< episode)
        gamma (float): Discount factor, same letter in the GAE paper
        lam (float): GAE parameter, 'lambda' in the GAE paper
        rew_key (str): Key associated with the reward entity
    """
    # Extract segment length
    length = len(segment[rew_key])
    # Augment the predicted values with the last predicted one (last not included in segment)
    # If the last `done` is 1, `segment['next_v']` is 0 (see segment generator)
    vs = np.append(segment["vs"], segment["next_v"])
    # Mark the last transition as nonterminal dones[length+1] = 0
    # If this is wrong and length+1 is in fact terminal (<=> dones[length+1] = 1),
    # then vs[length+1] = 0 by definition (see segment generator), so when we later
    # treat t = length in the loop, setting dones[length+1] = 0 leads to the correct values,
    # whether length+1 is terminal or non-terminal.
    dones = np.append(segment["dones"], 0)

    # Create empty GAE-modded advantage vector (same length as the reward minibatch)
    rews = segment[rew_key]
    gae_advs = np.empty_like(rews, dtype='float32')

    last_gae_adv = 0
    # Using a reversed loop naturally stacks the powers of gamma * lambda by wrapping
    # e.g. gae_rews[T-3] = delta[T-2] + gamma * lambda * (delta[T-1] + gamma * lambda * delta[T])
    # = delta[T-2] + gamma * lambda * delta[T-1] + (gamma * lambda)**2 * delta[T]
    # The computed GAE advantage relies only on deltas of future timesteps, hence the reversed
    for t in reversed(range(length)):
        # Wether the current transition is terminal
        nonterminal = 1 - dones[t + 1]
        # Compute the 1-step Temporal Difference residual
        delta = rews[t] + (gamma * vs[t + 1] * nonterminal) - vs[t]
        # Compute the GAE-modded advantage and add it to the advantage vector
        last_gae_adv = delta + (gamma * lam * nonterminal * last_gae_adv)
        gae_advs[t] = last_gae_adv

    # Augment the segment with the constructed statistics
    segment["advs"] = gae_advs  # vector containing the GAE advantages
    # Add the values (baselines) to the advantages to get the returns (MC Q)
    segment["td_lam_rets"] = gae_advs + segment["vs"]


def conjugate_gradient(f_Ax, b, cg_iters=50, residual_tol=1e-10, verbose=False):
    """ Conjugate gradient algorithm (Demmel, page 312)"""
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    if verbose:
        _tuples = []

    for i in range(cg_iters):
        if verbose:
            _tuples.append((i, rdotr, np.linalg.norm(x)))
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    if verbose:
        _tuples.append((i + 1, rdotr, np.linalg.norm(x)))
        print(columnize(names=["iter", "residual norm", "soln norm"],
                        tuples=_tuples,
                        widths=[4, 16, 16]))

    return x


def explained_variance(ypred, y):
    """Computes fraction of variance that 'ypred' explains about 'y'
    Returns 1 - Var[y-ypred] / Var[y]

    Interpretation:
        ev=0 => might as well have predicted zero
        ev=1 => perfect prediction
        ev<0 => worse than just predicting zero
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    out = 1 - np.var(y - ypred) / vary
    return np.nan if vary == 0 else out
