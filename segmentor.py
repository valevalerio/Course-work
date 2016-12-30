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
                left_hist, right_hist = itemfreq(left_labels), itemfreq(right_labels)
                cur_impurity = impurity(left_hist, right_hist)
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

class Gini:

    def __call__(self, left_label_hist, right_label_hist):
        left_bincount, right_bincount = left_label_hist[:,1], right_label_hist[:,1]
        left_total, right_total = np.sum(left_bincount), np.sum(right_bincount)

        left_entropy = self._cal_gini(left_bincount, left_total)
        right_entropy = self._cal_gini(right_bincount, right_total)

        total = left_total + right_total

        return (left_total/total) * left_entropy + (right_total/total) * right_entropy

    def _cal_gini(self, bincount, total):
        gini = 1.0
        for count in bincount:
            freq = count/total
            gini -= freq**2
        return gini
