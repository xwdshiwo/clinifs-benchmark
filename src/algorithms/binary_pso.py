import numpy as np
import time
from .base import BaseFeatureSelector


class BinaryPSO(BaseFeatureSelector):
    """
    Binary PSO for feature selection.
    Velocity: real-valued; position: binary via sigmoid + Bernoulli sampling.
    Fitness: alpha * error_rate + beta * feature_ratio (lower is better).
    """

    def __init__(self, n_particles=20, max_iter=100, w=0.9, c1=2.0, c2=2.0,
                 alpha=0.9, beta=0.1, k_neighbors=5, random_state=None, verbose=True):
        super().__init__(n_particles=n_particles, max_iter=max_iter,
                         k_neighbors=k_neighbors, alpha=alpha, beta=beta,
                         random_state=random_state, verbose=verbose)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    @staticmethod
    def _sigmoid(v):
        return 1.0 / (1.0 + np.exp(-np.clip(v, -500, 500)))

    def fit(self, X, y):
        start = time.time()
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # Initialize binary positions and real velocities
        pos = rng.randint(0, 2, (self.n_particles, n_features)).astype(float)
        vel = rng.uniform(-1, 1, (self.n_particles, n_features))

        # Evaluate initial fitness
        fitness = np.array([self._fitness_function(X, y, pos[i].astype(bool))
                            for i in range(self.n_particles)])

        pbest_pos = pos.copy()
        pbest_fit = fitness.copy()

        gbest_idx = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]

        fitness_curve = np.zeros(self.max_iter + 1)
        num_features_curve = np.zeros(self.max_iter + 1)
        fitness_curve[0] = gbest_fit
        num_features_curve[0] = np.sum(gbest_pos)

        for it in range(1, self.max_iter + 1):
            r1 = rng.rand(self.n_particles, n_features)
            r2 = rng.rand(self.n_particles, n_features)

            vel = (self.w * vel
                   + self.c1 * r1 * (pbest_pos - pos)
                   + self.c2 * r2 * (gbest_pos - pos))

            # Sigmoid transfer + Bernoulli sampling
            prob = self._sigmoid(vel)
            pos = (rng.rand(self.n_particles, n_features) < prob).astype(float)

            for i in range(self.n_particles):
                f = self._fitness_function(X, y, pos[i].astype(bool))
                if f < pbest_fit[i]:
                    pbest_fit[i] = f
                    pbest_pos[i] = pos[i].copy()
                if f < gbest_fit:
                    gbest_fit = f
                    gbest_pos = pos[i].copy()

            fitness_curve[it] = gbest_fit
            num_features_curve[it] = np.sum(gbest_pos)

            if self.verbose:
                print(f"Iter {it}/{self.max_iter} - Fitness: {gbest_fit:.6f} - "
                      f"Features: {int(num_features_curve[it])}")

        self.best_features_ = np.where(gbest_pos.astype(bool))[0]
        self.selected_features_ = self.best_features_
        self.best_fitness_ = gbest_fit
        self.fitness_curve_ = fitness_curve
        self.num_features_curve_ = num_features_curve
        self.execution_time_ = time.time() - start
        return self
