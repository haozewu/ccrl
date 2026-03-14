from __future__ import annotations

import random

from sklearn.utils import shuffle


class ImbalanceCrossValidation:
    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def _split_once(self, idx, dataset):
        test_index_list = []
        train_index_list = []
        for single_collection in dataset:
            test_index = random.sample(range(0, len(single_collection)), self.n_splits)
            train_index = [i for i in range(len(single_collection)) if i not in test_index]
            test_index_list.append(test_index)
            train_index_list.append(shuffle(train_index))
        return idx, train_index_list, test_index_list

    def split(self, dataset):
        random.seed(self.random_state)
        for idx in range(self.n_repeats):
            yield self._split_once(idx, dataset)
