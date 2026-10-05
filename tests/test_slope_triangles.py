"""Slope triangles: a rate only where the error follows a power law."""
import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cavsim2d.utils.slope_triangles import add_slope_triangles, fit_convergence_rate

N = np.array([100, 300, 1000, 3000, 10000, 30000])


def test_a_power_law_gives_its_rate():
    fit = fit_convergence_rate(N, 5.0 * N ** -3.0)
    assert fit is not None and fit[0] == pytest.approx(3.0)


def test_round_off_points_are_left_out_of_the_fit():
    y = 5.0 * N ** -3.0
    y[-2:] = 1e-15                      # hit the floor
    fit = fit_convergence_rate(N, y, floor=5e-14)
    assert fit is not None and fit[0] == pytest.approx(3.0)


def test_a_zig_zag_or_a_rising_curve_gets_no_rate():
    zig = np.array([1e-3, 1e-4, 8e-4, 3e-3, 1e-4, 5e-4])
    assert fit_convergence_rate(N, zig) is None
    assert fit_convergence_rate(N, 1e-6 * N) is None
    assert fit_convergence_rate(N[:1], [1e-3]) is None


def test_triangles_are_drawn_only_for_power_laws():
    fig, ax = plt.subplots()
    good, zig = 5.0 * N ** -2.0, np.array([1e-3, 1e-4, 8e-4, 3e-3, 1e-4, 5e-4])
    ax.loglog(N, good)
    ax.loglog(N, zig)
    n_lines = len(ax.get_lines())
    rates = add_slope_triangles(ax, [(N, good), (N, zig)], colors=['C0', 'C1'])
    assert rates[0] == pytest.approx(2.0) and rates[1] is None
    assert len(ax.get_lines()) == n_lines + 1          # one triangle
    assert any(t.get_text() == '2.0' for t in ax.texts)
    plt.close(fig)
