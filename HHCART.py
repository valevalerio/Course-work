from sklearn.base import BaseEstimator
from scipy.linalg import norm
from sklearn.decomposition import PCA
import numpy as np

class HHCARTNode:

    def __init__(self, depth, labels, **kwargs):
        self.depth = depth
        self.labels = labels
        self.is_leaf = kwargs.get('is_leaf', False)
        self._split_rules = kwargs.get('split_rules', None)
        self._householder_matrix = kwargs.get('householder', None)
        self._left_child = kwargs.get('left_child', None)
        self._right_child = kwargs.get('right_child', None)

        if not self.is_leaf:
            assert self._split_rules
            assert self._left_child
            assert self._right_child

    def get_child(self, datum):
        if self.is_leaf:
            raise StandardError("Leaf node does not have children.")
        feature_index, threshhold = self.split_rules
        X = datum
        if self._householder_matrix is not None:
            X = X.dot(self._householder_matrix)
        
        if X[feature_index] < threshhold:
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

class HouseHolderCART(BaseEstimator):

    def __init__(self, impurity, segmentor, tau=1e-4, **kwargs):
        self.impurity = impurity
        self.segmentor = segmentor
        self.tau = tau
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
            node = HHCARTNode(cur_depth, y, is_leaf=True)
            self._nodes.append(node)
            return node
    
    def _generate_node(self, X, y, cur_depth):
        if self._terminate(X, y, cur_depth):
            return self._generate_leaf_node(cur_depth, y)
        else:
            n_objects, n_features = X.shape
            impurity_best, sr, left_indices, right_indices = self.segmentor(X, y, self.impurity)
            
            #generate Housholder matrix
            extractor = PCA(n_components=1)
            extractor.fit(X, y)
            mu = extractor.components_[0]
            
            I = np.diag(np.ones(n_features))
            check_ = np.sqrt((I - mu).T ** 2).sum(axis=0)
            if (check_ > self.tau).sum == n_features:
                i = np.where(check_ > self.tau)[0][0]
                e = np.zeros(n_features)
                e[i] = 1.0
                w = (e - mu) / norm(e - mu) 
                householder_matrix = I - 2 * w[:, np.newaxis].dot(w[:, np.newaxis].T)
                X_house = X.dot(householder_matrix)
                impurity_house, sr_house, left_indices_house, right_indices_house = self.segmentor(X_house, y, self.impurity)
                
                if impurity_best > impurity_house:
                    impurity_best = impurity_house
                    left_indices = left_indices_house
                    right_indices = right_indices_house
                    sr = sr_house
            else:
                householder_matrix = I
                
            if not sr:
                return self._generate_leaf_node(cur_depth, y)
                
            X_left, y_left = X[left_indices], y[left_indices]
            X_right, y_right = X[right_indices], y[right_indices]

            node = HHCARTNode(cur_depth, y,
                        split_rules=sr,
                        householder=householder_matrix,
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
