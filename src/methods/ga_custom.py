"""
Custom Genetic Algorithm with explicit sparsity penalty and max_features.
Replaces GAFeatureSelectionCV which lacks these controls.

Fitness = alpha * (1 - AUC_cv) + beta * (n_features / total)  [minimize]
"""
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


class GAFeatureSelector:
    def __init__(self, alpha=0.5, beta=0.5, pop_size=30, n_gen=20,
                 max_features=None, mutation_rate=0.05, crossover_rate=0.8,
                 tournament_size=3, elite_size=2,
                 random_state=42, verbose=False, cv=3):
        self.alpha = alpha
        self.beta  = beta
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.max_features = max_features
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.random_state = random_state
        self.verbose = verbose
        self.cv = cv

        self.best_features_ = None
        self.best_fitness_ = None
        self.fitness_curve_ = None
        self.n_features_curve_ = None
        self.feature_frequencies_ = None
        self.execution_time_ = None

    def _fitness(self, X, y, chrom):
        if chrom.sum() == 0:
            return 1.0
        X_sel = X[:, chrom.astype(bool)]
        clf = LogisticRegression(C=1, solver="liblinear",
                                 max_iter=300, random_state=42)
        try:
            scores = cross_val_score(clf, X_sel, y, cv=self.cv,
                                     scoring="roc_auc", n_jobs=1)
            auc = float(np.mean(scores))
        except Exception:
            auc = 0.5
        return self.alpha * (1 - auc) + self.beta * (chrom.sum() / len(chrom))

    def _enforce_max(self, chrom, rng):
        if self.max_features is None:
            return chrom
        ones = np.where(chrom == 1)[0]
        if len(ones) <= self.max_features:
            return chrom
        # Keep random subset of max_features
        keep = rng.choice(ones, size=self.max_features, replace=False)
        new = np.zeros_like(chrom)
        new[keep] = 1
        return new

    def _init_population(self, n_features, rng):
        pop = []
        # Initialize with ~target_density ones
        if self.max_features is not None:
            target = self.max_features
        else:
            target = max(10, n_features // 20)
        for _ in range(self.pop_size):
            chrom = np.zeros(n_features, dtype=int)
            n_ones = rng.randint(max(3, target // 2), max(target + 1, 6))
            idx = rng.choice(n_features, size=min(n_ones, n_features), replace=False)
            chrom[idx] = 1
            pop.append(chrom)
        return np.array(pop)

    def _tournament(self, pop, fit, rng):
        candidates = rng.choice(len(pop), size=self.tournament_size, replace=False)
        best = candidates[np.argmin(fit[candidates])]
        return pop[best].copy()

    def _crossover(self, p1, p2, rng):
        if rng.rand() > self.crossover_rate:
            return p1.copy(), p2.copy()
        # Uniform crossover
        mask = rng.rand(len(p1)) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        return c1, c2

    def _mutate(self, chrom, rng):
        # Bit-flip mutation at low rate
        flip = rng.rand(len(chrom)) < self.mutation_rate
        new = chrom.copy()
        new[flip] = 1 - new[flip]
        return new

    def fit(self, X, y):
        t0 = time.time()
        n_features = X.shape[1]
        rng = np.random.RandomState(self.random_state)

        pop = self._init_population(n_features, rng)
        pop = np.array([self._enforce_max(c, rng) for c in pop])
        fit_arr = np.array([self._fitness(X, y, c) for c in pop])

        self.fitness_curve_ = np.zeros(self.n_gen + 1)
        self.n_features_curve_ = np.zeros(self.n_gen + 1)
        self.fitness_curve_[0] = fit_arr.min()
        self.n_features_curve_[0] = pop[np.argmin(fit_arr)].sum()

        for gen in range(1, self.n_gen + 1):
            # Elitism
            elite_idx = np.argsort(fit_arr)[:self.elite_size]
            new_pop = [pop[i].copy() for i in elite_idx]

            # Fill rest via selection + crossover + mutation
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(pop, fit_arr, rng)
                p2 = self._tournament(pop, fit_arr, rng)
                c1, c2 = self._crossover(p1, p2, rng)
                c1 = self._mutate(c1, rng)
                c2 = self._mutate(c2, rng)
                c1 = self._enforce_max(c1, rng)
                c2 = self._enforce_max(c2, rng)
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            pop = np.array(new_pop[:self.pop_size])
            fit_arr = np.array([self._fitness(X, y, c) for c in pop])

            self.fitness_curve_[gen] = fit_arr.min()
            self.n_features_curve_[gen] = pop[np.argmin(fit_arr)].sum()

            if self.verbose:
                print(f"Gen {gen}/{self.n_gen}  Fit={fit_arr.min():.4f}  "
                      f"Feat={int(self.n_features_curve_[gen])}")

        best_idx = np.argmin(fit_arr)
        self.best_features_ = np.where(pop[best_idx] == 1)[0]
        self.best_fitness_ = float(fit_arr[best_idx])
        # Feature frequency across final population
        self.feature_frequencies_ = pop.sum(axis=0) / self.pop_size
        self.execution_time_ = time.time() - t0
        return self
