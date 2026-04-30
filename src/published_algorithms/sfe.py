"""
SFE (Simple, Fast, and Efficient) Feature Selection Algorithm

Reference
---------
Ahadzadeh, B., Abdar, M., Safara, F., Khosravi, A., Men Haj, M. B., &
Suganthan, P. N. (2023). SFE: A Simple, Fast, and Efficient Feature
Selection Algorithm for High-Dimensional Data. IEEE Transactions on
Evolutionary Computation. DOI: 10.1109/TEVC.2023.3238420
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from .base import BaseFeatureSelector


class SFE(BaseFeatureSelector):
    """
    SFE (Simple, Fast, and Efficient) Feature Selection Algorithm

    A simple yet efficient feature selection algorithm that uses two operators:
    1. Nonselection Operator: Randomly unselects features to explore the search space
    2. Selection Operator: Ensures at least one feature is always selected

    The algorithm uses a dynamically decreasing Unselection Rate (UR) that starts
    high for global exploration and gradually decreases for local exploitation.

    Parameters
    ----------
    max_iter : int, default=60
        Maximum number of iterations (function evaluations)

    ur_max : float, default=0.3
        Maximum unselection rate (between 0 and 1)
        Higher values mean more features are unselected in early iterations

    ur_min : float, default=0.001
        Minimum unselection rate (between 0 and 1)
        Lower bound for the unselection rate

    k_neighbors : int, default=5
        Number of neighbors for KNN classifier used in fitness function

    random_state : int or None, default=None
        Random seed for reproducibility

    verbose : bool, default=True
        Whether to print progress information

    Attributes
    ----------
    best_solution_ : ndarray of shape (n_features,)
        Binary mask of selected features (1=selected, 0=not selected)

    best_features_ : ndarray
        Indices of selected features

    best_fitness_ : float
        Best fitness (accuracy) achieved

    convergence_curve_ : list
        Fitness values at each iteration

    n_selected_features_ : int
        Number of selected features in the best solution

    Examples
    --------
    >>> from fslib.algorithms import SFE
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> sfe = SFE(max_iter=30, random_state=42)
    >>> sfe.fit(X, y)
    >>> X_selected = sfe.transform(X)
    >>> print(f"Selected {sfe.n_selected_features_} features")
    """

    def __init__(
        self,
        max_iter=60,
        ur_max=0.3,
        ur_min=0.001,
        k_neighbors=5,
        random_state=None,
        verbose=True
    ):
        # Input validation
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError(f"max_iter must be a positive integer, got {max_iter}")

        if not (0 < ur_max <= 1):
            raise ValueError(f"ur_max must be in (0, 1], got {ur_max}")

        if not (0 < ur_min < ur_max):
            raise ValueError(f"ur_min must be in (0, ur_max), got {ur_min}")

        if not isinstance(k_neighbors, int) or k_neighbors < 1:
            raise ValueError(f"k_neighbors must be a positive integer, got {k_neighbors}")

        self.max_iter = max_iter
        self.ur_max = ur_max
        self.ur_min = ur_min
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.verbose = verbose

        # Initialize random number generator
        self.rng = np.random.RandomState(random_state)

        # Results (will be set during fit)
        self.best_solution_ = None
        self.best_features_ = None
        self.best_fitness_ = None
        self.convergence_curve_ = []
        self.n_selected_features_ = None

    def fit(self, X, y):
        """
        Execute SFE algorithm to find optimal feature subset

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : SFE
            Fitted estimator
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"SFE Algorithm - Feature Selection")
            print(f"{'='*70}")
            print(f"Dataset: {n_samples} samples, {n_features} features")
            print(f"Parameters: max_iter={self.max_iter}, UR=[{self.ur_min}, {self.ur_max}]")
            print(f"{'='*70}\n")

        # Algorithm 1: SFE Feature Selection

        # Step 1: Initialize an individual X (binary encoding)
        X_solution = self.rng.randint(0, 2, n_features)

        # Ensure at least one feature is selected initially
        if np.sum(X_solution) == 0:
            X_solution[self.rng.randint(0, n_features)] = 1

        # Step 2: Calculate the fitness of X
        fitness_X = self._fitness_function(X, y, X_solution)

        # Initialize best solution
        self.best_solution_ = X_solution.copy()
        self.best_fitness_ = fitness_X
        self.convergence_curve_ = []

        # Current unselection rate
        ur = self.ur_max

        # Step 3: Main loop
        for iteration in range(1, self.max_iter + 1):
            # Create a copy for new solution
            X_new = X_solution.copy()

            # ===================================================================
            # Nonselection Operator (Exploration Phase)
            # ===================================================================
            # Find indices of selected features (where X == 1)
            selected_indices = np.where(X_solution == 1)[0]
            n_selected = len(selected_indices)

            # Calculate the number of features to unselect: UN = ceil(UR × nvar)
            un = int(np.ceil(ur * n_features))  # Equation (1) in paper

            # Randomly select UN features from the currently selected features
            if n_selected > 0 and un > 0:
                # Generate UN random indices (with possible duplicates)
                k_indices = self.rng.randint(0, n_selected, un)
                # Remove duplicates
                k_indices_unique = np.unique(k_indices)
                # Get the actual feature indices to unselect
                features_to_unselect = selected_indices[k_indices_unique]
                # Unselect these features
                X_new[features_to_unselect] = 0

            # ===================================================================
            # Selection Operator (Exploitation Phase)
            # ===================================================================
            # If all features are unselected, randomly select one feature
            if np.sum(X_new) == 0:
                # Find indices of non-selected features (where X == 0)
                non_selected_indices = np.where(X_solution == 0)[0]
                n_non_selected = len(non_selected_indices)

                if n_non_selected > 0:
                    # Randomly select 1 feature to select (SN = 1)
                    sn = 1
                    k_index = self.rng.randint(0, n_non_selected, sn)
                    feature_to_select = non_selected_indices[k_index[0]]
                    # Reset to original solution and select this feature
                    X_new = X_solution.copy()
                    X_new[feature_to_select] = 1
                else:
                    # Edge case: all features were originally selected
                    # Just randomly select one feature
                    X_new[self.rng.randint(0, n_features)] = 1

            # ===================================================================
            # Fitness Evaluation and Update
            # ===================================================================
            # Calculate the fitness of X_new
            fitness_X_new = self._fitness_function(X, y, X_new)

            # Greedy selection: accept if better
            if fitness_X_new > fitness_X:
                X_solution = X_new.copy()
                fitness_X = fitness_X_new

                # Update best solution
                if fitness_X > self.best_fitness_:
                    self.best_solution_ = X_solution.copy()
                    self.best_fitness_ = fitness_X

            # Record convergence
            self.convergence_curve_.append(self.best_fitness_)

            # ===================================================================
            # Update Unselection Rate (UR)
            # ===================================================================
            # UR = (URmax - URmin) × ((Max_FEs - FEs) / Max_FEs) + URmin
            # Equation (2) in paper - UR decreases linearly over iterations
            ur = (self.ur_max - self.ur_min) * ((self.max_iter - iteration) / self.max_iter) + self.ur_min

            # Print progress
            if self.verbose:
                print(f"Iter {iteration:3d}/{self.max_iter}: "
                      f"Accuracy={fitness_X:.4f}, "
                      f"Best={self.best_fitness_:.4f}, "
                      f"Features={np.sum(X_solution):4d}/{n_features}, "
                      f"UR={ur:.4f}")

        # Store final results
        self.best_features_ = np.where(self.best_solution_ == 1)[0]
        self.n_selected_features_ = len(self.best_features_)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"SFE Algorithm Completed")
            print(f"{'='*70}")
            print(f"Best Accuracy: {self.best_fitness_:.4f}")
            print(f"Selected Features: {self.n_selected_features_}/{n_features} "
                  f"({100 * (1 - self.n_selected_features_/n_features):.2f}% reduction)")
            print(f"{'='*70}\n")

        return self

    def _fitness_function(self, X, y, feature_mask):
        """
        Fitness function based on KNN classification accuracy

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,)
            Target values

        feature_mask : array-like of shape (n_features,)
            Binary mask (1=selected, 0=not selected)

        Returns
        -------
        fitness : float
            Classification accuracy (between 0 and 1)
        """
        # Get selected features
        selected_features = np.where(feature_mask == 1)[0]

        # Handle edge case: no features selected
        if len(selected_features) == 0:
            return 0.0

        # Extract selected features
        X_selected = X[:, selected_features]

        # Use KNN classifier with cross-validation
        knn = KNeighborsClassifier(n_neighbors=self.k_neighbors)

        try:
            # 5-fold cross-validation (as in paper)
            cv_scores = cross_val_score(knn, X_selected, y, cv=5, scoring='accuracy')
            fitness = np.mean(cv_scores)
        except Exception:
            # Fallback if cross-validation fails
            fitness = 0.0

        return fitness

    def transform(self, X):
        """
        Transform X to selected features

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        Returns
        -------
        X_selected : array of shape (n_samples, n_selected_features)
            Data with only selected features
        """
        if self.best_features_ is None:
            raise RuntimeError("SFE must be fitted before transform")

        X = np.asarray(X)
        return X[:, self.best_features_]

    def fit_transform(self, X, y):
        """
        Fit SFE and transform X

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        X_selected : array of shape (n_samples, n_selected_features)
            Transformed data with only selected features
        """
        return self.fit(X, y).transform(X)
