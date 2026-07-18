"""cavsim2d house plotting style — a warm-cast, publication-oriented palette.

The goal is one visual identity across every figure the toolkit draws. Import
what you need:

- ``WARM``             the qualitative palette (ordered, print-safe hues). It is
  *warm-cast categorical, alternating*: consecutive entries jump cool<->warm, so
  any two adjacent categories are maximally distinct.
- ``polarisation_color(pol)``  a stable hue per polarisation (monopole -> steel
  teal, dipole -> rust, ...), so the same physical quantity keeps its colour.
- ``shades(color, n)`` ``n`` tints of one hue, base first — for series that
  belong together (e.g. the successive passbands of one polarisation). Light
  bases darken and dark bases lighten, so every shade stays legible.
- ``cmap(name)``       a warm continuous colormap for field / density plots.
- ``apply_style()`` and the ``house_style()`` context manager set matplotlib
  rcParams (fonts, colour cycle, subtle grid) without a LaTeX installation.

The palette stays legible in greyscale because the hues also differ in lightness.
"""
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np

from cavsim2d.solvers.eigenmode_result import pol_number

# Warm-cast categorical palette, alternating cool <-> warm so neighbours in the
# cycle contrast strongly. Built out from the Persian-green / sandy-brown /
# burnt-sienna family.
WARM = [
    '#457B9D',   #  0 steel teal
    '#C1440E',   #  1 rust
    '#2A9D8F',   #  2 teal
    '#F4A261',   #  3 marigold
    '#264653',   #  4 deep teal
    '#E76F51',   #  5 vermillion
    '#6FA88F',   #  6 seafoam
    '#E9C46A',   #  7 gold
    '#5B7553',   #  8 moss
    '#8A9A5B',   #  9 olive
    '#9E2A2B',   # 10 crimson
    '#EE964B',   # 11 amber
    '#3D5A45',   # 12 forest
    '#8B4000',   # 13 brick
]

#: A warm continuous colormap (deep ember -> straw), for fields / densities.
WARM_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'warm', ['#3B1710', '#7A2A1E', '#C25128', '#E0A23A', '#F5E1A4'])

try:                                    # register so plt.get_cmap('warm') works
    mpl.colormaps.register(WARM_CMAP, force=True)
except (AttributeError, ValueError):    # older mpl, or already registered
    pass


def polarisation_color(pol):
    """A stable hue for a polarisation ('dipole', 1, ...)."""
    return WARM[pol_number(pol) % len(WARM)]


def shades(color, n):
    """``n`` shades of *color*, base first (``shades(c, k)[0] == c``).

    Used to colour series that belong to one hue — e.g. the successive passbands
    of a polarisation. Light bases darken and dark bases lighten (always toward
    the far end of the lightness range), so a dark hue like deep teal does not
    turn to mud and every shade stays distinct.
    """
    rgb = np.array(mcolors.to_rgb(color))
    if n <= 1:
        return [mcolors.to_hex(rgb)]
    luminance = float(np.dot(rgb, (0.2126, 0.7152, 0.0722)))
    target = rgb * 0.5 if luminance > 0.5 else rgb + (1.0 - rgb) * 0.6
    return [mcolors.to_hex(np.clip((1 - t) * rgb + t * target, 0, 1))
            for t in np.linspace(0.0, 1.0, n)]


def cmap(_name='warm'):
    """The house continuous colormap."""
    return WARM_CMAP


# House font. STIX is the font family the APS / AIP physics journals use
# (Physical Review Accelerators and Beams among them), and — the reason it is
# chosen here — its mathtext set matches the text exactly, so the pi-fraction
# tick labels render in the *same* font as the legend and axis labels. It ships
# with matplotlib, so no LaTeX install and no external font are needed.
HOUSE_FONT = 'STIXGeneral'
HOUSE_MATHFONT = 'stix'

#: rcParams for the house style. No usetex; mathtext renders ``$\frac{\pi}{2}$``.
_RC = {
    'axes.prop_cycle': mpl.cycler(color=WARM),
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#222222',
    'axes.linewidth': 0.9,
    'axes.grid': True,
    'grid.color': '#D9D2C7',
    'grid.linewidth': 0.6,
    'grid.alpha': 0.6,
    'axes.axisbelow': True,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'figure.facecolor': 'white',
    'font.family': 'serif',
    'font.serif': [HOUSE_FONT, 'DejaVu Serif'],
    'font.size': 11,
    'legend.frameon': False,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
    'mathtext.fontset': HOUSE_MATHFONT,   # matches HOUSE_FONT; no LaTeX needed
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}


def set_font(text=None, math=None):
    """Change the house font. ``text`` is a font family (e.g. 'Palatino
    Linotype', 'Georgia'); ``math`` is a mathtext set for the equations and the
    fraction tick labels ('stix', 'cm' for the LaTeX look, 'dejavuserif', ...).

    For the tick labels and legend to share one font, keep ``math`` matched to
    ``text`` — 'stix' pairs with STIXGeneral, 'cm' with a Computer Modern text
    font. Applies to subsequent :func:`apply_style` / :class:`house_style` use.
    """
    if text is not None:
        _RC['font.family'] = 'serif'
        _RC['font.serif'] = [text, 'DejaVu Serif']
    if math is not None:
        _RC['mathtext.fontset'] = math


def apply_style():
    """Set the house style globally (``matplotlib.rcParams``)."""
    mpl.rcParams.update(_RC)


class house_style:
    """Context manager applying the house style to the enclosed plotting only.

    Non-invasive: leaves the caller's global rcParams untouched.
    """

    def __enter__(self):
        self._ctx = mpl.rc_context(_RC)
        return self._ctx.__enter__()

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)
