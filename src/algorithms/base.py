from abc import ABC, abstractmethod
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


class BaseFeatureSelector(ABC):
    def __init__(self, n_particles=20, max_iter=100, k_neighbors=5,
                 alpha=0.9, beta=0.1, random_state=None, verbose=True):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state
        self.verbose = verbose
        self.best_features_ = None
        self.best_fitness_ = None
        self.fitness_curve_ = None
        self.num_features_curve_ = None
        self.execution_time_ = None
        if random_state is not None:
            np.random.seed(random_state)

    @abstractmethod
    def fit(self, X, y):
        pass

    def transform(self, X):
        if self.best_features_ is None:
            raise ValueError("Call fit() first")
        return X[:, self.best_features_]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def _fitness_function(self, X, y, feature_mask):
        if not np.any(feature_mask):
            return 1.0
        X_sel = X[:, feature_mask]
        knn = KNeighborsClassifier(n_neighbors=self.k_neighbors)
        scores = cross_val_score(knn, X_sel, y, cv=5, scoring='accuracy')
        error_rate = 1 - np.mean(scores)
        feature_ratio = np.sum(feature_mask) / len(feature_mask)
        return self.alpha * error_rate + self.beta * feature_ratio
