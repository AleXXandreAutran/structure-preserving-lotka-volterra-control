from fractions import Fraction
import numpy as np
from reduced_global import isolate, phi0_bounds, evaluate, sturm, variations


def test_exact_sturm_count():
    roots, seq = isolate([-2, 0, 1], Fraction(0), Fraction(3))
    assert len(roots) == 1
    a, b = roots[0]
    assert a * a < 2 < b * b
    assert variations(sturm([1, 0, 1]), Fraction(0)) - variations(sturm([1, 0, 1]), Fraction(3)) == 0


def test_global_candidate_separation():
    roots, _ = isolate([2, -5, -3, 1, 1], Fraction(0), Fraction(3))
    intervals = [(Fraction(0), Fraction(0)), *roots, (Fraction(3), Fraction(3))]
    bounds = [phi0_bounds(*i) for i in intervals]
    assert len(roots) == 2
    assert all((bounds[2][1] < lo for k, (lo, _) in enumerate(bounds) if k != 2))


def test_flux_hessian_positive_and_old_alternating_negative():
    for x, z in [(0.0, 0.0), (0.7, 0.7), (1.8, 0.7), (3.0, 3.0)]:
        a, b = (1 + x, 1 + z)
        H = np.array([
            [1 + 4 / (a ** 3 * b), 2 / (a * a * b * b)],
            [2 / (a * a * b * b), 1 + 4 / (a * b ** 3)],
        ])
        assert np.linalg.eigvalsh(H).min() >= 1
    H0 = np.array([[1, 2], [2, 1]])
    d = np.array([1.0, -1.0]) / np.sqrt(2)
    assert abs(d @ H0 @ d + 1) < 1e-14
