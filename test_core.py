import numpy as np
from aqol import AQOL


def test_converges_toward_minimum_1d():
    def f(x):
        return (x - 1.0) ** 2  # minimum connu en x=1

    np.random.seed(0)
    opt = AQOL(bounds=(-5, 5), random_state=0)
    for _ in range(25):
        x = opt.ask()
        opt.tell(x, f(x) + np.random.normal(0, 0.05))

    x_best, _ = opt.best()
    assert isinstance(x_best, float)
    assert abs(x_best - 1.0) < 0.5


def test_ask_stays_within_bounds_1d():
    opt = AQOL(bounds=(-2, 2), random_state=1)
    for _ in range(15):
        x = opt.ask()
        assert -2 <= x <= 2
        opt.tell(x, x ** 2)


def test_multidim_converges():
    def f(x):
        return (x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2  # minimum en (1, -2)

    np.random.seed(0)
    opt = AQOL(bounds=[(-5, 5), (-5, 5)], random_state=0)
    for _ in range(40):
        x = opt.ask()
        opt.tell(x, f(x) + np.random.normal(0, 0.05))

    x_best, _ = opt.best()
    assert x_best.shape == (2,)
    assert abs(x_best[0] - 1.0) < 0.7
    assert abs(x_best[1] + 2.0) < 0.7


def test_multidim_stays_within_bounds():
    opt = AQOL(bounds=[(-1, 1), (0, 10)], random_state=2)
    for _ in range(20):
        x = opt.ask()
        assert -1 <= x[0] <= 1 and 0 <= x[1] <= 10
        opt.tell(x, x[0] ** 2 + x[1])
