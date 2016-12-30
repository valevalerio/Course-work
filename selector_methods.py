import numpy as np
from sklearn.neighbors import KDTree
from sklearn.feature_selection import f_classif


class MutualInformation():
    def __init__(self, 
                 continuous_variables='two_continuous'):
        self.continuous_variables = continuous_variables
    
    def mutual_info(self, x, y):
        if (self.continuous_variables == 'two_continuous'):
            hx, bx = histogram(x, bins = int(np.sqrt(x.shape[0])), density=True)
            x_categorical = digitize(x, bx)
            hy, by = histogram(y, bins = int(np.sqrt(y.shape[0])), density=True)
            y_categorical = digitize(y, by)
            return mutual_info_score(x_categorical, y_categorical)
        else:
            return self.entropy(y) - self.conditional_entropy(x, y)
    
    def entropy(self, y):
        
        p_y = self.compute_distribution(y)
            
        self.entropy_ = 0.0
        for k, v in p_y.items():
            self.entropy_ += v * log2(v)
        self.entropy_ = -self.entropy_
        
        return self.entropy_

    def conditional_entropy(self, x, y):
        hx, bx = histogram(x, bins = int(np.sqrt(x.shape[0])), density=True)
        x_categorical = digitize(x, bx)
        p_x = self.compute_distribution(x_categorical)
        
        y_categorical = y
        p_y = self.compute_distribution(y)
        
        self.cond_entropy_ = 0.0
        tmp = 0.0
        for y_ in set(y_categorical):
            x1 = x[y_categorical == y_]
            cond_p_xy = self.compute_distribution(digitize(x1, bx)) #p(x/ y)
            for k, v in cond_p_xy.items():
                self.cond_entropy_ += (v * p_y[y_] * (log2(p_x[k]) - log2(v * p_y[y_])))
        return self.cond_entropy_
    
    def compute_distribution(self, x):
        d = defaultdict(int)
        for e in x: d[e] += 1
        s = float(sum(d.values()))
        a = dict((k, x / s) for k, x in d.items())
        return a


class Selector():
    
    def __init__(self, criterion='correlation'):
        self.criterion = criterion
    
    def fit(self, X, y):
        
        def mutual_scores(self, X, y):
            scores_ = np.zeros(X.shape[1])
            m = MutualInformation(continuous_variables='one_continuous')

            for i in range(X.shape[1]):
                x = X[:, i]
                scores_[i] = m.mutual_info(x, y)

            return scores_
        
        def correlation_coef(X, y):
            return np.mean((X - X.mean(axis=0)) * (X - y.mean()), axis=0) \
                            / np.sqrt(X.std(axis=0) * y.std())
        
        def relief_criterion(X, y):
    
            num_classes = np.unique(y)
            
            k = 5
            
            for c in num_classes:
                x_c = X[y == c]
                k = min(k, x_c.shape[0] - 1)
            
            same_class = list()
            different_class = list()
            for c in num_classes:
                x_c = X[y == c]
                tree = KDTree(x_c)
                inds_c = np.where(y == c)[0]
                nearest_neighbor = tree.query(x_c, k=k + 1, return_distance=False)[:, 1:]
                same_class.append((inds_c, inds_c[nearest_neighbor]))
                  
                x_not_c = X[y != c]
                inds_c = np.where(y != c)[0]
                tree = KDTree(x_not_c)
                nearest_neighbor = tree.query(x_c, k=k + 1, return_distance=False)[:, 1:]
                different_class.append((inds_c, inds_c[nearest_neighbor]))
                
            scores_ = np.zeros(X.shape[1])    
                
            for k_ in range(k):
                for c in num_classes:
                    x_c = X[y == c]
                    ind_diff = different_class[c][1][:, k_]
                    ind_same = same_class[c][1][:, k_]
                    scores_ += (np.abs(x_c - X[ind_diff]) / \
                               (np.abs(x_c - X[ind_same]) + 1e-10)).sum(axis=0)
            return scores_
        
        if (self.criterion == 'correlation'):
            self.scores_ = np.abs(correlation_coef(X, y))
        
        if (self.criterion == 'mutual_info'):
            self.scores_ = self.mutual_scores(X, y)
        
        if (self.criterion == 'f_classif'):
            self.scores_ = f_classif(X, y)[0]
            
        if (self.criterion == 'relief_criterion'):
            self.scores_ = relief_criterion(X, y)
        
        return self