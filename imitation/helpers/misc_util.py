import random

import numpy as np
import tensorflow as tf


def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def intprod(x):
    return int(np.prod(x))


def numel(x):
    """Returns the number of elements in `x`"""
    return intprod(var_shape(x))


def onehotify(targets, num_dims):
    """Transform an array of discrete actions
    into an array containing their hot encodings

    Example:
        action `2` or `2.`, with a total of 4 possible actions -> [0., 0., 1., 0.] (1-dim array)
    """
    assert targets.ndim == 2, "minibatch dimension must be 2"
    targets = targets.reshape(-1).astype(dtype=np.int64)
    one_hot_targets = np.eye(num_dims)[targets]
    assert (np.sum(one_hot_targets, axis=-1).reshape(-1) == 1.).all(), "encoding failed"
    return one_hot_targets


def flatten_lists(listoflists):
    """Flatten a list of lists"""
    return [el for list_ in listoflists for el in list_]


def zipsame(*seqs):
    """Verify that all the sequences in `seqs` are the same length, then zip them together"""
    assert seqs, "empty input sequence"
    ref_len = len(seqs[0])
    assert all(len(seq) == ref_len for seq in seqs[1:])
    return zip(*seqs)


def unpack(seq, sizes):
    """Unpack `seq` into a sequence of lists, with lengths specified by `sizes`.
    `None` in `sizes` means just one bare element, not a list.

    Example:
        unpack([1, 2, 3, 4, 5, 6], [3, None, 2]) -> ([1, 2, 3], 4, [5, 6])

    Technically `upack` returns a generator object, i.e. an iterator over ([1, 2, 3], 4, [5, 6])
    """
    seq = list(seq)
    it = iter(seq)
    assert sum(1 if s is None else s for s in sizes) == len(seq), \
        "Trying to unpack %s into %s" % (seq, sizes)
    for size in sizes:
        if size is None:
            yield it.__next__()
        else:
            li = []
            for _ in range(size):
                li.append(it.__next__())
            yield li


def set_global_seeds(i):
    """Set global seeds"""
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def prettify_time(seconds):
    """Print the number of seconds in human-readable format.

    Examples:
        '2 days', '2 hours and 37 minutes', 'less than a minute'.
    """
    minutes = seconds // 60
    seconds %= 60
    hours = minutes // 60
    minutes %= 60
    days = hours // 24
    hours %= 24

    def helper(count, name):
        return "{} {}{}".format(str(count), name, ('s' if count > 1 else ''))

    # Display only the two greater units (days and hours, hours and minutes, minutes and seconds)
    if days > 0:
        message = helper(days, 'day')
        if hours > 0:
            message += ' and ' + helper(hours, 'hour')
        return message
    if hours > 0:
        message = helper(hours, 'hour')
        if minutes > 0:
            message += ' and ' + helper(minutes, 'minute')
        return message
    if minutes > 0:
        return helper(minutes, 'minute')
    # Finally, if none of the previous conditions is valid
    return 'less than a minute'


def boolean_flag(parser, name, default=False, help=None):
    """Add a boolean flag to argparse parser.

    Args:
        parser (argparse.Parser): Parser to add the flag to
        name (str): Name of the flag
          --<name> will enable the flag, while --no-<name> will disable it
        default (bool or None): Default value of the flag
        help (str): Help string for the flag
    """
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)
