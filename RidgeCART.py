from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
import numpy as np
from numpy.linalg import norm

class Node:

    def __init__(self, depth, labels, **kwargs):
        self.depth = depth
        self.labels = labels
        self.is_leaf = kwargs.get('is_leaf', False)
        self._split_rules = kwargs.get('split_rules', None)
        self._method = kwargs.get('method', 'usual')
        self._weights = kwargs.get('weights', None)
        self._left_child = kwargs.get('left_child', None)
        self._right_child = kwargs.get('right_child', None)

        if not self.is_leaf:
            assert self._split_rules
            assert self._left_child
            assert self._right_child

    def get_child(self, datum):
        if self.is_leaf:
            raise StandardError("Leaf node does not have children.")
        i, treshold = self._split_rules
        
        if self._method == 'usual':
            X_hat = np.zeros(datum.shape[0] + 1)
            X_hat[:-1] = datum
            X_hat[-1] = self._weights.dot(datum)
        elif self._method == 'hard':
            X_hat = np.array([self._weights.dot(datum)])

        if X_hat[i] < treshold:
            return self.left_child
        else:
            return self.right_child

    @property
    def label(self):
        if not hasattr(self, '_label'):
            self._label = np.mean(self.labels)
        return self._label

    @property
    def split_rules(self):
        if self.is_leaf:
            raise StandardError("Leaf node does not have split rule.")
        return self._split_rules

    @property
    def left_child(self):
        if self.is_leaf:
            raise StandardError("Leaf node does not have split rule.")
        return self._left_child

    @property
    def right_child(self):
        if self.is_leaf:
            raise StandardError("Leaf node does not have split rule.")
        return self._right_child


class Ridge_CART(BaseEstimator):

    def __init__(self, impurity, segmentor, alpha=1.0, method='usual', **kwargs):
        self.impurity = impurity
        self.segmentor = segmentor
        self.alpha = alpha
        self.method = method # hard and usual
        self._max_depth = kwargs.get('max_depth', None)
        self._min_samples = kwargs.get('min_samples', 2)
        self._root = None
        self._nodes = []
    
    def _terminate(self, X, y, cur_depth):
            if self._max_depth != None and cur_depth == self._max_depth:
                # maximum depth reached.
                return True
            elif y.size < self._min_samples:
                # minimum number of samples reached.
                return True
            elif np.unique(y).size == 1:
                return True
            else:
                return False
    
    def _generate_leaf_node(self, cur_depth, y):
            node = Node(cur_depth, y, is_leaf=True)
            self._nodes.append(node)
            return node
    
    def _generate_node(self, X, y, cur_depth):
        if self._terminate(X, y, cur_depth):
            return self._generate_leaf_node(cur_depth, y)
        else:
            n_objects, n_features = X.shape
            
            regressor = Ridge(alpha=self.alpha)
            regressor.fit(X, y)
            weights = regressor.coef_
            weights = weights / norm(weights)
            
            if self.method == 'usual':
                X_hat = np.hstack((X, X.dot(weights)[:, np.newaxis]))
            elif self.method == 'hard':
                X_hat = X.dot(weights)[:, np.newaxis]
            
            impurity, sr, left_indices, right_indices = self.segmentor(X_hat, y, self.impurity)
            
            if not sr:
                return self._generate_leaf_node(cur_depth, y)
            
            X_left, y_left = X[left_indices], y[left_indices]
            X_right, y_right = X[right_indices], y[right_indices]

            node = Node(cur_depth, y,
                        split_rules=sr,
                        method=self.method,
                        weights=weights,
                        left_child=self._generate_node(X_left, y_left, cur_depth + 1),
                        right_child=self._generate_node(X_right, y_right, cur_depth + 1),
                        is_leaf=False)
            self._nodes.append(node)
            return node
    
    def fit(self, X, y):
        
        self._root = self._generate_node(X, y, 0)

    def predict(self, X):
        def predict_single(datum):
            cur_node = self._root
            while not cur_node.is_leaf:
                cur_node = cur_node.get_child(datum)
            return cur_node.label
        
        if not self._root:
            raise StandardError("Decision tree has not been trained.")
        size = X.shape[0]
        predictions = np.empty((size, ), dtype=float)
        for i in range(size): 
            predictions[i] = predict_single(X[i, :])
        return predictions

    def score(self, data, labels):
        if not self._root:
            raise StandardError("Decision tree has not been trained.")
        predictions = self.predict(data)
        correct_count = np.count_nonzero(predictions == labels)
        return correct_count / labels.shape[0]
