#!/usr/bin/env python3
"""
Abstract base class shared by the supplementary feature-selection algorithms
(MEL, SFE) in this repository.
"""
from abc import ABC, abstractmethod
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import time


class BaseFeatureSelector(ABC):
    """
    Abstract base class for population-based feature-selection algorithms.

    Subclasses must implement ``fit`` and may reuse the shared
    ``transform`` / ``fit_transform`` / ``_fitness_function`` helpers.
    """

    def __init__(self,
                 n_particles=20,
                 max_iter=100,
                 k_neighbors=5,
                 alpha=0.9,
                 beta=0.1,
                 random_state=None,
                 verbose=True):
        """
        Initialise shared search parameters.

        Parameters
        ----------
        n_particles : int, default=20
            Population (or swarm) size.
        max_iter : int, default=100
            Maximum number of search iterations.
        k_neighbors : int, default=5
            Number of neighbours used by the internal KNN classifier in
            the fitness function.
        alpha : float, default=0.9
            Weight on the classification error term in the fitness
            function.
        beta : float, default=0.1
            Weight on the panel-size penalty term in the fitness
            function.
        random_state : int, optional
            Random seed for reproducibility.
        verbose : bool, default=True
            If True, log per-iteration progress.
        """
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state
        self.verbose = verbose

        # Runtime trace populated during ``fit``.
        self.best_features_ = None
        self.best_fitness_ = None
        self.fitness_curve_ = None
        self.num_features_curve_ = None
        self.execution_time_ = None

        if random_state is not None:
            np.random.seed(random_state)

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the feature-selection model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : object
            Returns self to support method chaining.
        """
        pass

    def transform(self, X):
        """
        Subset the input matrix to the selected features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix to transform.

        Returns
        -------
        X_selected : ndarray of shape (n_samples, n_selected_features)
            Matrix restricted to the indices stored in ``best_features_``.
        """
        if self.best_features_ is None:
            raise ValueError("Model has not been fitted; call fit() first.")

        return X[:, self.best_features_]

    def fit_transform(self, X, y):
        """
        Fit the selector and return the transformed matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        X_selected : ndarray of shape (n_samples, n_selected_features)
            Matrix restricted to the selected features.
        """
        return self.fit(X, y).transform(X)

    def _fitness_function(self, X, y, feature_mask):
        """
        Compute the fitness value for a candidate subset.

        Parameters
        ----------
        X : ndarray
            Feature matrix.
        y : ndarray
            Labels.
        feature_mask : ndarray of bool
            Boolean mask indicating selected features (True = selected).

        Returns
        -------
        fitness : float
            Fitness value; lower is better.
        """
        # Empty subset gets the worst possible fitness.
        if not np.any(feature_mask):
            return 1.0

        # Apply the mask to keep only the selected features.
        X_selected = X[:, feature_mask]

        # 5-fold cross-validation with a KNN classifier as the inner model.
        knn = KNeighborsClassifier(n_neighbors=self.k_neighbors)
        scores = cross_val_score(knn, X_selected, y, cv=5, scoring='accuracy')
        accuracy = np.mean(scores)
        error_rate = 1 - accuracy

        # Panel-size ratio relative to the full feature pool.
        n_selected = np.sum(feature_mask)
        n_total = len(feature_mask)
        feature_ratio = n_selected / n_total

        # Fitness = alpha * error rate + beta * panel-size ratio.
        fitness = self.alpha * error_rate + self.beta * feature_ratio

        return fitness

    def get_results(self):
        """
        Collect the run summary.

        Returns
        -------
        results : dict
            Dictionary with the selected feature indices, best fitness
            value, convergence curves, and execution time.
        """
        if self.best_features_ is None:
            raise ValueError("Model has not been fitted.")

        return {
            'best_features': self.best_features_,
            'n_selected_features': len(self.best_features_),
            'best_fitness': self.best_fitness_,
            'best_accuracy': 1 - self.best_fitness_,  # Approximate accuracy.
            'fitness_curve': self.fitness_curve_,
            'num_features_curve': self.num_features_curve_,
            'execution_time': self.execution_time_
        }
