"""
Deux exemples : optimisation à 1 paramètre (API scalaire) et à plusieurs
paramètres (API vectorielle), comparés à une recherche aléatoire.

Lancer avec :  python examples/optimize_demo.py
"""
import numpy as np
from aqol import AQOL

NOISE_STD = 0.2
BUDGET_1D = 30
BUDGET_ND = 60


# ---------- Exemple 1 : un seul paramètre ----------
def hidden_cost_1d(x: float) -> float:
    return np.sin(2 * x) + 0.5 * np.sin(4 * x) + 0.05 * x ** 2


def measure_1d(x: float) -> float:
    return hidden_cost_1d(x) + np.random.normal(0, NOISE_STD)


def run_1d(strategy: str) -> float:
    if strategy == "aqol":
        opt = AQOL(bounds=(-3.2, 3.2))
        for _ in range(BUDGET_1D):
            x = opt.ask()
            opt.tell(x, measure_1d(x))
        x_best, _ = opt.best()
    else:
        xs = np.random.uniform(-3.2, 3.2, size=BUDGET_1D)
        ys = [measure_1d(x) for x in xs]
        x_best = xs[int(np.argmin(ys))]
    return hidden_cost_1d(x_best)


# ---------- Exemple 2 : trois paramètres ----------
def hidden_cost_3d(x: np.ndarray) -> float:
    a, b, c = x
    return (a - 1) ** 2 + np.sin(2 * b) + 0.3 * (c + 2) ** 2


def measure_3d(x: np.ndarray) -> float:
    return hidden_cost_3d(x) + np.random.normal(0, NOISE_STD)


def run_3d(strategy: str) -> float:
    bounds = [(-3, 3), (-3, 3), (-3, 3)]
    if strategy == "aqol":
        opt = AQOL(bounds=bounds)
        for _ in range(BUDGET_ND):
            x = opt.ask()
            opt.tell(x, measure_3d(x))
        x_best, _ = opt.best()
    else:
        low, high = np.array([-3, -3, -3]), np.array([3, 3, 3])
        xs = np.random.uniform(low, high, size=(BUDGET_ND, 3))
        ys = [measure_3d(x) for x in xs]
        x_best = xs[int(np.argmin(ys))]
    return hidden_cost_3d(x_best)


if __name__ == "__main__":
    print("=== 1 paramètre ===")
    print(f"AQOL   : coût réel = {run_1d('aqol'):.4f}")
    print(f"Random : coût réel = {run_1d('random'):.4f}")

    print("\n=== 3 paramètres ===")
    print(f"AQOL   : coût réel = {run_3d('aqol'):.4f}")
    print(f"Random : coût réel = {run_3d('random'):.4f}")
