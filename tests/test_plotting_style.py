"""The house plotting style and the dispersion axis labelling — pure functions,
no solver needed."""
import matplotlib.colors as mcolors
import pytest

from cavsim2d.utils.style import WARM, polarisation_color, shades, house_style


def test_polarisation_colours_are_stable_and_distinct():
    mono, dip, quad = (polarisation_color(p) for p in ('monopole', 'dipole', 'quadrupole'))
    assert mono == polarisation_color(0)          # name or number
    assert dip == polarisation_color(1)
    assert len({mono, dip, quad}) == 3            # distinct hues
    assert all(c in WARM for c in (mono, dip, quad))


def test_palette_alternates_cool_and_warm():
    """The palette is warm-cast *alternating*: neighbours contrast because one
    leans warm (red > blue) and the next leans cool (blue >= red)."""
    warmth = [mcolors.to_rgb(c)[0] - mcolors.to_rgb(c)[2] for c in WARM]  # r - b
    # monopole (0) is cool, dipole (1) is warm -> strong monopole/dipole contrast
    assert warmth[0] < 0 < warmth[1]
    # neighbours alternate sign for the first several entries
    signs = [w > 0 for w in warmth[:6]]
    assert all(signs[i] != signs[i + 1] for i in range(len(signs) - 1))


def test_shades_start_at_base_and_stay_monotonic():
    for base in ('#F4A261', '#264653'):           # a light hue and a dark hue
        out = shades(base, 4)
        assert out[0].lower() == mcolors.to_hex(mcolors.to_rgb(base)).lower()
        lums = [sum(mcolors.to_rgb(c)) for c in out]
        # a dark base lightens, a light base darkens — monotonic either way
        assert lums == sorted(lums) or lums == sorted(lums, reverse=True)
        assert len(set(out)) == len(out)          # all shades distinct
    assert shades('#264653', 1) == [mcolors.to_hex(mcolors.to_rgb('#264653'))]


def test_house_style_does_not_leak_globally():
    import matplotlib as mpl
    before = mpl.rcParams['axes.prop_cycle']
    with house_style():
        pass
    assert mpl.rcParams['axes.prop_cycle'] == before


@pytest.mark.parametrize('j, n, expected', [
    (1, 2, r'$\dfrac{\pi}{2}$'),
    (2, 2, r'$\pi$'),                # (n/n) reduces to pi, not (2/2)pi
    (2, 4, r'$\dfrac{\pi}{2}$'),     # 2/4 -> 1/2
    (6, 9, r'$\dfrac{2\pi}{3}$'),    # 6/9 -> 2/3
    (9, 9, r'$\pi$'),
])
def test_phase_advance_label_reduces_fractions(j, n, expected):
    # imported lazily: importing the model package pulls in ngsolve-adjacent code
    pytest.importorskip('ngsolve')
    from cavsim2d.models.base import Cavity
    assert Cavity._phase_advance_label(j, n) == expected


def test_resolve_bands_flat_list_applies_to_every_polarisation():
    pytest.importorskip('ngsolve')
    from cavsim2d.models.base import Cavity
    assert Cavity._resolve_bands(None, ['monopole', 'dipole']) == [None, None]
    assert Cavity._resolve_bands([1, 2], ['monopole', 'dipole']) == [[1, 2], [1, 2]]


def test_resolve_bands_nested_list_is_per_polarisation():
    pytest.importorskip('ngsolve')
    from cavsim2d.models.base import Cavity
    # [[1, 2], [1]] with pol=[monopole, dipole]: two bands for monopole, one for dipole
    assert Cavity._resolve_bands([[1, 2], [1]], ['monopole', 'dipole']) == [[1, 2], [1]]


def test_resolve_bands_rejects_length_mismatch_and_mixed_forms():
    pytest.importorskip('ngsolve')
    from cavsim2d.models.base import Cavity
    with pytest.raises(ValueError, match='per-polarisation'):
        Cavity._resolve_bands([[1, 2], [1]], ['monopole'])          # 2 lists, 1 pol
    with pytest.raises(ValueError, match='mixes'):
        Cavity._resolve_bands([1, [2]], ['monopole', 'dipole'])     # mixed forms


def test_auto_clustering_splits_far_apart_passbands():
    pytest.importorskip('ngsolve')
    import numpy as np
    from cavsim2d.models.base import Cavity
    # the user's case: ~1300, 1620-1870, ~2400 — two real gaps
    vals = np.concatenate([np.linspace(1290, 1320, 9), np.linspace(1620, 1785, 9),
                           np.linspace(1845, 1870, 3), np.linspace(2380, 2420, 3)])
    clusters = Cavity._frequency_clusters(vals)
    assert len(clusters) == 3
    assert clusters[0] == pytest.approx((1290, 1320))
    assert clusters[-1] == pytest.approx((2380, 2420))


def test_manual_breaks_place_the_splits_by_hand():
    pytest.importorskip('ngsolve')
    import numpy as np
    from cavsim2d.models.base import Cavity
    vals = np.concatenate([np.linspace(1290, 1320, 9), np.linspace(1620, 1785, 9),
                           np.linspace(1845, 1870, 3), np.linspace(2380, 2420, 3)])
    # one gap at ~1450 splits into exactly two clusters; approximate bounds are fine
    two = Cavity._clusters_from_breaks(vals, [(1350, 1600)])
    assert len(two) == 2
    assert two[0][1] == pytest.approx(1320) and two[1][0] == pytest.approx(1620)
    # two gaps -> three clusters (what the user expected)
    three = Cavity._clusters_from_breaks(vals, [(1350, 1600), (1900, 2350)])
    assert len(three) == 3
    with pytest.raises(ValueError, match='low, high'):
        Cavity._clusters_from_breaks(vals, [1500])          # not a (low, high) pair


def test_phase_advances_span_zero_to_pi():
    """The passband phase advances run 0 (0-mode) to pi (pi-mode) at q*pi/(n-1),
    not the old j*pi/n that omitted the 0-mode."""
    pytest.importorskip('ngsolve')
    import numpy as np
    from cavsim2d.models.base import Cavity
    mu, labels = Cavity._phase_advances(9)
    assert mu[0] == 0.0 and mu[-1] == pytest.approx(np.pi)
    assert labels[0] == r'$0$' and labels[-1] == r'$\pi$'
    assert labels[4] == r'$\dfrac{\pi}{2}$'          # q=4 of 8 -> pi/2
    assert Cavity._phase_advances(1)[1] == [r'$\pi$']   # single cell -> pi-mode
