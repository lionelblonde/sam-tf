import numpy as np


class Feeder(object):

    def __init__(self, data_map, enable_shuffle=True):
        """Initiate a Feeder object, iteratable data feed dictionary"""
        self.data_map = data_map  # to differentiate from 'feed_dict'
        self.enable_shuffle = enable_shuffle
        self.n = next(iter(data_map.values())).shape[0]
        self.next_index = 0
        if self.enable_shuffle:
            self.shuffle()

    def shuffle(self):
        perm = np.arange(self.n)
        np.random.shuffle(perm)
        for key in self.data_map:
            self.data_map[key] = self.data_map[key][perm]
        self.next_index = 0

    def get_next_batch(self, batch_size):
        if self.next_index >= self.n and self.enable_shuffle:
            self.shuffle()
        cur_index = self.next_index
        cur_batch_size = min(batch_size, self.n - self.next_index)
        self.next_index += cur_batch_size
        batch = dict()
        for key in self.data_map:
            batch[key] = self.data_map[key][cur_index:cur_index + cur_batch_size]
        return batch

    def get_feed(self, batch_size):
        if self.enable_shuffle:
            self.shuffle()
        while self.next_index <= self.n - batch_size:
            yield self.get_next_batch(batch_size)
        self.next_index = 0

    def extract(self, m):
        """Extract the `m` first elements for each item in the data map"""
        sub_data_map = dict()
        for key in self.data_map:
            sub_data_map[key] = self.data_map[key][:m]
        return Feeder(sub_data_map)
