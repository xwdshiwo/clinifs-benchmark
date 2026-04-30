import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.datasets import make_classification
from algorithms import BinaryPSO

X, y = make_classification(n_samples=100, n_features=50, random_state=42)

print("=== BinaryPSO Smoke Test ===")
pso = BinaryPSO(n_particles=20, max_iter=5, random_state=42, verbose=True)
pso.fit(X, y)

print(f"\nSelected features: {len(pso.selected_features_)}/50")
print(f"Best fitness: {pso.best_fitness_:.6f}")
print("PASSED")
