"""
AQOL — Adaptive Quantum-classical Optimization Loop
Boucle d'optimisation adaptative pour fonctions coûteuses et bruitées à mesurer
(mesures quantiques, calibration physique, essais coûteux, etc.), à un ou
plusieurs paramètres.

Repose sur des principes établis d'optimisation bayésienne (processus gaussien +
amélioration espérée). AQOL assemble ces briques sous une API simple dédiée au
cas "chaque mesure coûte cher".
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from dataclasses import dataclass, field


def _rbf_kernel(A: np.ndarray, B: np.ndarray, length_scale: float, signal_var: float) -> np.ndarray:
    """A: (n, d), B: (m, d) -> matrice (n, m)."""
    sq_dist = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
    return signal_var * np.exp(-sq_dist / (2.0 * length_scale ** 2))


@dataclass
class _GaussianProcess:
    length_scale: float = 0.5
    signal_var: float = 1.0
    noise_var: float = 1e-2
    X: np.ndarray = field(default_factory=lambda: np.empty((0, 1)))
    y: np.ndarray = field(default_factory=lambda: np.empty(0))
    _L: np.ndarray | None = field(default=None, repr=False)
    _alpha: np.ndarray | None = field(default=None, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X, self.y = X, y
        n = len(X)
        K = _rbf_kernel(X, X, self.length_scale, self.signal_var)
        K += np.eye(n) * (self.noise_var + 1e-8)
        # Cholesky pour stabilité numérique (au lieu de inv)
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))

    def predict(self, X_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._L is None or self._alpha is None:
            raise RuntimeError("GP non entraîné. Appeler fit() d'abord.")
        k_star = _rbf_kernel(X_star, self.X, self.length_scale, self.signal_var)
        mean = k_star @ self._alpha
        v = np.linalg.solve(self._L, k_star.T)
        var = self.signal_var - np.sum(v ** 2, axis=0)
        return mean, np.sqrt(np.clip(var, 1e-12, None))

    def log_marginal_likelihood(self) -> float:
        """Log-vraisemblance marginale (pour optimisation des hyperparamètres)."""
        if self._L is None or self._alpha is None:
            return -np.inf
        n = len(self.y)
        data_fit = -0.5 * self.y @ self._alpha
        log_det = np.sum(np.log(np.diag(self._L)))
        return float(data_fit - log_det - 0.5 * n * np.log(2 * np.pi))


def _expected_improvement(mean: np.ndarray, sd: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    """EI pour un problème de MINIMISATION."""
    improvement = best - mean - xi
    z = np.divide(improvement, sd, out=np.zeros_like(sd), where=sd > 1e-9)
    ei = improvement * norm.cdf(z) + sd * norm.pdf(z)
    return np.where(sd > 1e-9, ei, 0.0)


def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating))


class AQOL:
    """
    Boucle d'optimisation adaptative pour une fonction coûteuse/bruitée à minimiser,
    à un ou plusieurs paramètres.

    Usage à 1 paramètre (API scalaire) :
        opt = AQOL(bounds=(-3, 3))
        x = opt.ask()                # float
        opt.tell(x, mesurer(x))
        x_best, estimation = opt.best()

    Usage à N paramètres :
        opt = AQOL(bounds=[(-3, 3), (0, 10), (-1, 1)])
        x = opt.ask()                 # np.ndarray de forme (3,)
        opt.tell(x, mesurer(*x))
        x_best, estimation = opt.best()
    """

    def __init__(
        self,
        bounds,
        n_init: int | None = None,
        n_candidates: int | None = None,
        length_scale: float = 0.5,
        signal_var: float = 1.2,
        noise_var: float = 1e-2,
        xi: float = 0.01,
        adapt_hyperparams: bool = True,
        random_state: int | None = None,
    ):
        if _is_number(bounds[0]) and _is_number(bounds[1]) and len(bounds) == 2:
            self._bounds = [(float(bounds[0]), float(bounds[1]))]
            self._scalar_interface = True
        else:
            self._bounds = [(float(b[0]), float(b[1])) for b in bounds]
            self._scalar_interface = False

        self.dim = len(self._bounds)
        self.low = np.array([b[0] for b in self._bounds])
        self.high = np.array([b[1] for b in self._bounds])

        # Plus de dimensions => plus de points d'amorçage et de candidats
        self.n_init = n_init if n_init is not None else max(3, 2 * self.dim)
        self.n_candidates = n_candidates if n_candidates is not None else min(5000, 400 * self.dim + 300)

        self.gp = _GaussianProcess(length_scale, signal_var, noise_var)
        self.xi = float(xi)
        self.adapt_hyperparams = bool(adapt_hyperparams)
        self.rng = np.random.default_rng(random_state)
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._sobol = Sobol(d=self.dim, scramble=True, seed=random_state)
        self._n_asks = 0

    def _out(self, point: np.ndarray):
        return float(point[0]) if self._scalar_interface else point.copy()

    def _sample_candidates(self, n: int) -> np.ndarray:
        """Candidats quasi-aléatoires (Sobol) projetés dans les bornes."""
        u = self._sobol.random(n)
        return self.low + u * (self.high - self.low)

    def _adapt_hyperparams(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Adaptation légère des hyperparamètres par grille (rapide, stable).
        Cherche length_scale et signal_var qui maximisent la log-vraisemblance.
        """
        if len(y) < 4:
            return

        length_scales = np.geomspace(0.05, 3.0, 8)
        signal_vars = np.geomspace(0.1, 5.0, 6)
        best_ll = -np.inf
        best_ls, best_sv = self.gp.length_scale, self.gp.signal_var

        for ls in length_scales:
            for sv in signal_vars:
                self.gp.length_scale = float(ls)
                self.gp.signal_var = float(sv)
                try:
                    self.gp.fit(X, y)
                    ll = self.gp.log_marginal_likelihood()
                    if ll > best_ll:
                        best_ll = ll
                        best_ls, best_sv = ls, sv
                except np.linalg.LinAlgError:
                    continue

        self.gp.length_scale = float(best_ls)
        self.gp.signal_var = float(best_sv)

    def ask(self):
        """Retourne le prochain point à mesurer."""
        self._n_asks += 1

        if len(self._X) < self.n_init:
            # Phase d'exploration initiale : Sobol pour bon remplissage
            point = self._sample_candidates(1)[0]
            return self._out(point)

        X = np.array(self._X)
        y = np.array(self._y)

        # Adaptation des hyperparamètres de temps en temps (coût faible)
        if self.adapt_hyperparams and (len(self._X) == self.n_init or self._n_asks % 5 == 0):
            self._adapt_hyperparams(X, y)

        self.gp.fit(X, y)

        candidates = self._sample_candidates(self.n_candidates)
        mean, sd = self.gp.predict(candidates)
        ei = _expected_improvement(mean, sd, best=float(y.min()), xi=self.xi)
        point = candidates[int(np.argmax(ei))]
        return self._out(point)

    def tell(self, x, y: float) -> None:
        """Enregistre le résultat d'une mesure. x : float (1D) ou séquence (nD)."""
        self._X.append(np.atleast_1d(np.asarray(x, dtype=float)))
        self._y.append(float(y))

    def best(self):
        """
        Meilleur point trouvé, estimé via la moyenne du modèle de substitution
        (et non la valeur brute bruitée) pour éviter le biais optimiste du bruit.
        """
        if len(self._X) < 2:
            i = int(np.argmin(self._y))
            return self._out(self._X[i]), self._y[i]

        X = np.array(self._X)
        y = np.array(self._y)
        self.gp.fit(X, y)
        mean, _ = self.gp.predict(X)
        i = int(np.argmin(mean))
        return self._out(X[i]), float(mean[i])

    def history(self):
        """Retourne (liste des points mesurés, liste des valeurs observées)."""
        return [self._out(x) for x in self._X], list(self._y)

    @property
    def n_observations(self) -> int:
        return len(self._X)
