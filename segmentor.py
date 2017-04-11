from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import itemfreq

class SegmentorBase:
    __metaclass__ = ABCMeta

    @abstractmethod
    def _split_generator(self, data):
        pass

    def __init__(self, msl=1):
        self._min_samples_leaf = msl

    def __call__(self, data, labels, impurity):
        best_impurity = float('inf')
        best_split_rule = None
        best_left_i = None
        best_right_i = None
        splits = self._split_generator(data)

        for left_i, right_i, split_rule in splits:
            if left_i.size > self._min_samples_leaf and right_i.size > self._min_samples_leaf:
                left_labels, right_labels = labels[left_i], labels[right_i]
                cur_impurity = impurity(left_labels, right_labels)
                if cur_impurity < best_impurity:
                    best_impurity = cur_impurity
                    best_split_rule = split_rule
                    best_left_i = left_i
                    best_right_i = right_i
        return (best_impurity,
                best_split_rule,
                best_left_i,
                best_right_i
        )

# Split based on mean value of each feature.
class MeanSegmentor(SegmentorBase):
    def _split_generator(self, data):
        for feature_i in range(data.shape[1]):
            feature_values = data[:,feature_i]
            mean = np.mean(feature_values)
            left_i = np.nonzero(feature_values < mean)[0]
            right_i = np.nonzero(feature_values >= mean)[0]
            split_rule = (feature_i, mean)
            yield (
                    left_i,
                    right_i,
                    split_rule
                    )

class MSE:

    def __call__(self, left_label, right_label):
        left_len, right_len = len(left_label), len(right_label)

        left_std = np.std(left_label)
        right_std = np.std(right_label)

        total = left_len + right_len

        return (left_len / total) * left_std + (right_len / total) * right_std