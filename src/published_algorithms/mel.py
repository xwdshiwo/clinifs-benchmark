#!/usr/bin/env python3
"""
MEL: Multi-Task Evolutionary Learning for Feature Selection.

Reference
---------
Wang, X., et al. (2024). MEL: Efficient Multi-Task Evolutionary Learning
for High-Dimensional Feature Selection. IEEE Transactions on Knowledge
and Data Engineering. DOI: 10.1109/TKDE.2024.3366333.

Upstream implementation: https://github.com/wangxb96/MEL
"""
import numpy as np
import time
from .base import BaseFeatureSelector


class MEL(BaseFeatureSelector):
    """
    MEL (Multi-Task Evolutionary Learning) feature selection.

    Core ideas
    ----------
    1. Two-subpopulation multi-task learning:
       - Sub1: standard PSO with cross-subpopulation guidance from Sub2.
       - Sub2: probabilistic selection driven by learned feature weights.
    2. Dynamic feature-weight learning: weights are updated according to
       the change in fitness between consecutive iterations.
    3. Knowledge transfer: Sub1 borrows information from Sub2's best
       solution, while Sub2 searches using weights shared with Sub1.

    Parameters
    ----------
    n_particles : int, default=20
        Population (swarm) size.
    max_iter : int, default=100
        Maximum number of iterations.
    c1 : float, default=2.0
        Cognitive coefficient (individual learning rate).
    c2 : float, default=2.0
        Social coefficient (global learning rate).
    c3 : float, default=2.0
        Cross-subpopulation learning rate.
    w : float, default=0.9
        Inertia weight.
    threshold : float, default=0.6
        Binarisation threshold used to convert continuous positions
        into a feature-selection mask.
    k_neighbors : int, default=5
        Number of KNN neighbours used by the inner fitness function.
    alpha : float, default=0.9
        Weight on the classification error term in the fitness function.
    beta : float, default=0.1
        Weight on the panel-size penalty term in the fitness function.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=True
        If True, log per-iteration progress.
    """

    def __init__(self,
                 n_particles=20,
                 max_iter=100,
                 c1=2.0,
                 c2=2.0,
                 c3=2.0,
                 w=0.9,
                 threshold=0.6,
                 k_neighbors=5,
                 alpha=0.9,
                 beta=0.1,
                 random_state=None,
                 verbose=True):

        super().__init__(
            n_particles=n_particles,
            max_iter=max_iter,
            k_neighbors=k_neighbors,
            alpha=alpha,
            beta=beta,
            random_state=random_state,
            verbose=verbose
        )

        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.w = w
        self.threshold = threshold

    def fit(self, X, y):
        """
        Fit the MEL feature-selection model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        start_time = time.time()

        n_samples, n_features = X.shape
        n_particles = self.n_particles
        max_iter = self.max_iter

        # Search-space configuration.
        lb, ub = 0, 1
        vmax = (ub - lb) / 2

        # Initialise population positions and velocities.
        positions = np.random.uniform(lb, ub, (n_particles, n_features))
        velocities = np.zeros((n_particles, n_features))

        # Feature-weight vector shared across the population.
        feature_weights = np.zeros(n_features)

        # Fitness storage.
        fitness = np.zeros(n_particles)
        best_global_fitness = np.inf

        # Two subpopulations as defined by MEL.
        n_subpops = 2
        subpop_size = n_particles // n_subpops
        subpop_best_fitness = np.ones(n_subpops) * np.inf
        subpop_best_positions = np.zeros((n_subpops, n_features))

        # Initial evaluation pass.
        j = 0
        for i in range(n_particles):
            feature_mask = positions[i, :] > self.threshold
            fitness[i] = self._fitness_function(X, y, feature_mask)

            # Update subpopulation best.
            if fitness[i] < subpop_best_fitness[j]:
                subpop_best_positions[j, :] = positions[i, :]
                subpop_best_fitness[j] = fitness[i]

            # Advance subpopulation index every ``subpop_size`` particles.
            if (i + 1) % subpop_size == 0:
                j += 1

            # Update global best.
            if fitness[i] < best_global_fitness:
                best_global_position = positions[i, :].copy()
                best_global_fitness = fitness[i]

        # Personal bests.
        personal_best_positions = positions.copy()
        personal_best_fitness = fitness.copy()

        # Convergence trace.
        fitness_curve = np.zeros(max_iter + 1)
        num_features_curve = np.zeros(max_iter + 1)
        fitness_curve[0] = best_global_fitness
        num_features_curve[0] = np.sum(best_global_position > self.threshold)

        # Main loop.
        for iteration in range(1, max_iter + 1):
            k = 0  # Subpopulation index.

            for i in range(n_particles):
                # Subpopulation 1: standard PSO with cross-subpopulation learning.
                if k == 0:
                    for d in range(n_features):
                        r1, r2, r3 = np.random.rand(3)

                        # Velocity update (Eq. 5) including the c3 term that
                        # incorporates information from Sub2's best position.
                        velocities[i, d] = (
                            self.w * velocities[i, d] +
                            self.c1 * r1 * (personal_best_positions[i, d] - positions[i, d]) +
                            self.c2 * r2 * (best_global_position[d] - positions[i, d]) +
                            self.c3 * r3 * (subpop_best_positions[1, d] - positions[i, d])
                        )

                        # Velocity clipping.
                        velocities[i, d] = np.clip(velocities[i, d], -vmax, vmax)

                    # Position update.
                    positions[i, :] = positions[i, :] + velocities[i, :]
                    positions[i, :] = np.clip(positions[i, :], lb, ub)

                # Subpopulation 2: weight-driven probabilistic selection.
                else:
                    # Effective positive feature weights (Eq. 7).
                    valid_weights = feature_weights.copy()
                    valid_weights[valid_weights < 0] = 0
                    sum_weights = np.sum(valid_weights)

                    if sum_weights > 0:
                        for d in range(n_features):
                            # Probabilistic selection driven by weights (Eq. 8).
                            prob = np.random.rand()
                            feature_prob = valid_weights[d] / sum_weights

                            if feature_prob > prob:
                                positions[i, d] = 1
                            else:
                                positions[i, d] = 0
                    else:
                        # Random re-initialisation when no positive weights.
                        positions[i, :] = np.random.uniform(lb, ub, n_features)

                # Evaluate fitness.
                feature_mask_new = positions[i, :] > self.threshold
                fitness[i] = self._fitness_function(X, y, feature_mask_new)

                # Feature-weight update (Eqs. 3-4).
                feature_mask_old = personal_best_positions[i, :] > self.threshold
                self._update_feature_weights(
                    feature_weights,
                    feature_mask_old,
                    feature_mask_new,
                    personal_best_fitness[i],
                    fitness[i]
                )

                # Update personal best.
                if fitness[i] < personal_best_fitness[i]:
                    personal_best_positions[i, :] = positions[i, :].copy()
                    personal_best_fitness[i] = fitness[i]

                # Update subpopulation best.
                if fitness[i] < subpop_best_fitness[k]:
                    subpop_best_positions[k, :] = positions[i, :].copy()
                    subpop_best_fitness[k] = fitness[i]

                # Advance subpopulation index.
                if (i + 1) % subpop_size == 0:
                    k += 1

                # Update global best.
                if personal_best_fitness[i] < best_global_fitness:
                    best_global_position = personal_best_positions[i, :].copy()
                    best_global_fitness = personal_best_fitness[i]

            # Record convergence trace.
            fitness_curve[iteration] = best_global_fitness
            num_features_curve[iteration] = np.sum(best_global_position > self.threshold)

            if self.verbose:
                print(f"Iteration {iteration}/{max_iter} - "
                      f"Best Fitness: {best_global_fitness:.6f} - "
                      f"Features: {int(num_features_curve[iteration])}")

        # Persist results on the estimator.
        self.best_features_ = np.where(best_global_position > self.threshold)[0]
        self.best_fitness_ = best_global_fitness
        self.fitness_curve_ = fitness_curve
        self.num_features_curve_ = num_features_curve
        self.execution_time_ = time.time() - start_time

        return self

    def _update_feature_weights(self, weights, mask_old, mask_new, fitness_old, fitness_new):
        """
        Update the shared feature-weight vector in place.

        Following Eqs. 3-4 of the MEL paper:
        - If accuracy improves, weights of newly added features increase by
          Δacc and weights of removed features decrease by Δacc.
        - If accuracy degrades, the opposite update is applied.

        Parameters
        ----------
        weights : ndarray
            Feature-weight vector to update in place.
        mask_old : ndarray of bool
            Previous feature mask.
        mask_new : ndarray of bool
            New feature mask.
        fitness_old : float
            Previous fitness value.
        fitness_new : float
            New fitness value.
        """
        # Note: lower fitness is better, so improvement corresponds to a
        # positive delta_fitness.
        delta_fitness = fitness_old - fitness_new

        # Identify features that changed status between iterations.
        changed = np.logical_xor(mask_old, mask_new)
        emerged = changed & (mask_new > mask_old)  # Newly added features.
        disappeared = changed & (mask_old > mask_new)  # Removed features.

        if delta_fitness > 0:  # Accuracy improved (Case 1).
            weights[emerged] += delta_fitness
            weights[disappeared] -= delta_fitness
        else:  # Accuracy degraded (Case 2).
            weights[emerged] += delta_fitness  # delta_fitness < 0, so it decreases.
            weights[disappeared] -= delta_fitness  # Which increases the weight.
