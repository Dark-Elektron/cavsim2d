"""Polarisation-qualified objective names.

A QOI name alone is ambiguous: monopole and dipole ``qois.json`` share 18 key
names (``freq [MHz]``, ``R/Q [Ohm]``, ``G [Ohm]``, ...). So every UQ and
optimisation objective must say which polarisation it means, and may also say
which mode::

    'monopole:R/Q [Ohm]'        # the polarisation's primary mode of interest
    'dipole:2:freq [MHz]'       # dipole, mode 2 (1-based), from qois_all_modes

The separator is ``:``; no QOI name contains one. Parsing splits at most twice,
so the QOI name itself may contain anything (``R/Q_t [Ohm/m^(2(m-1))]``).

Unqualified names are rejected: ``'freq [MHz]'`` used to silently mean *the
monopole's*, which is a wrong answer waiting to happen once a second
polarisation is solved.
"""
import json
import os

from cavsim2d.solvers.eigenmode_result import pol_name, pol_number

SEP = ':'


class Objective:
    """A parsed objective: a polarisation, an optional 1-based mode, a QOI name."""

    __slots__ = ('pol', 'mode', 'qoi', 'text')

    def __init__(self, pol, mode, qoi, text):
        self.pol = pol          # canonical polarisation name, e.g. 'dipole'
        self.mode = mode        # 1-based mode index, or None for the primary
        self.qoi = qoi          # QOI key as it appears in qois.json
        self.text = text        # the original string

    @property
    def column(self):
        """Canonical column name for result tables."""
        if self.mode is None:
            return f'{self.pol}{SEP}{self.qoi}'
        return f'{self.pol}{SEP}{self.mode}{SEP}{self.qoi}'

    def __repr__(self):
        return f'Objective({self.column!r})'

    def __eq__(self, other):
        return isinstance(other, Objective) and self.column == other.column

    def __hash__(self):
        return hash(self.column)


def parse_objective(text):
    """``'dipole:2:freq [MHz]'`` -> :class:`Objective`.

    Raises ``ValueError`` on an unqualified name, an unknown polarisation, or a
    non-positive mode index.
    """
    if not isinstance(text, str):
        raise ValueError(f'objective must be a string, got {text!r}.')
    raw = text.strip()
    if SEP not in raw:
        raise ValueError(
            f"objective {text!r} does not name a polarisation. Monopole and m-pole "
            f"results share QOI names, so an objective must be written "
            f"'<polarisation>{SEP}<qoi>' (e.g. 'monopole{SEP}R/Q [Ohm]') or "
            f"'<polarisation>{SEP}<mode>{SEP}<qoi>' to pick a specific 1-based mode "
            f"(e.g. 'dipole{SEP}2{SEP}freq [MHz]').")

    parts = raw.split(SEP, 2)
    pol_token = parts[0].strip()
    try:
        # a bare azimuthal number ('1:freq [MHz]') is as valid as a name
        key = int(pol_token) if pol_token.lstrip('+').isdigit() else pol_token
        pol = pol_name(pol_number(key))
    except Exception:
        raise ValueError(
            f"objective {text!r} names an unknown polarisation {pol_token!r}. "
            f"Use a name ('monopole', 'dipole', 'quadrupole', ...) or its azimuthal "
            f"number (0, 1, 2, ...).") from None

    mode = None
    if len(parts) == 3:
        mode_token = parts[1].strip()
        if not mode_token.lstrip('+').isdigit():
            raise ValueError(
                f"objective {text!r}: the middle field must be a 1-based mode index, "
                f"got {mode_token!r}. Write '<pol>{SEP}<qoi>' for the primary mode of "
                f"interest, or '<pol>{SEP}<mode>{SEP}<qoi>' for a specific one.")
        mode = int(mode_token)
        if mode < 1:
            raise ValueError(
                f"objective {text!r}: mode indices are 1-based; mode 1 is the lowest of "
                f"the passband. Got {mode}.")
        qoi = parts[2].strip()
    else:
        qoi = parts[1].strip()

    if not qoi:
        raise ValueError(f'objective {text!r} has an empty QOI name.')
    return Objective(pol, mode, qoi, raw)


def is_qualified(text):
    """Whether *text* names a polarisation (and so is an eigenmode objective).

    Wakefield objectives (``'ZL'``, ``'ZT'``) carry no polarisation.
    """
    return isinstance(text, str) and SEP in text


def canonical_objective(text):
    """Normalise an eigenmode objective to its canonical column name.

    ``'1:freq [MHz]'`` -> ``'dipole:freq [MHz]'``. Non-eigenmode objectives
    (wakefield ZL/ZT) are returned unchanged.
    """
    if not is_qualified(text):
        return text
    return parse_objective(text).column


def parse_objectives(objectives):
    """Parse a list of objective strings, keeping order and rejecting duplicates."""
    out = []
    for text in objectives:
        obj = parse_objective(text)
        if obj in out:
            raise ValueError(f'objective {obj.column!r} is listed more than once.')
        out.append(obj)
    return out


def objective_polarisations(objectives):
    """Azimuthal mode numbers the objectives need, sorted (for eigenmode_config)."""
    return sorted({pol_number(o.pol) for o in parse_objectives(objectives)})


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def read_objective_values(eigenmode_dir, objectives):
    """Values of *objectives* from a cavity's ``eigenmode/`` directory.

    Returns ``{column: value}``. Raises ``ValueError`` naming the available QOIs
    if an objective's polarisation, mode or QOI is missing — previously an
    unmatched objective was silently dropped, so the statistics were quietly
    computed over whichever objectives happened to match.
    """
    values = {}
    for obj in parse_objectives(objectives):
        pol_dir = os.path.join(str(eigenmode_dir), obj.pol)

        if obj.mode is None:
            qois = _load(os.path.join(pol_dir, 'qois.json'))
            where = f"{obj.pol}/qois.json"
        else:
            all_modes = _load(os.path.join(pol_dir, 'qois_all_modes.json'))
            if all_modes is None:
                qois = None
            else:
                qois = all_modes.get(str(obj.mode - 1))     # file is 0-based keyed
                if qois is None:
                    raise ValueError(
                        f"objective {obj.text!r}: polarisation {obj.pol!r} has "
                        f"{len(all_modes)} solved modes, so mode {obj.mode} does not "
                        f"exist. Raise eigenmode_config['n_modes'].")
            where = f"{obj.pol}/qois_all_modes.json"

        if qois is None:
            raise ValueError(
                f"objective {obj.text!r}: no {where} under {eigenmode_dir}. Add "
                f"{obj.pol!r} to eigenmode_config['polarisation'] and rerun.")
        if obj.qoi not in qois:
            available = ', '.join(sorted(k for k, v in qois.items()
                                         if isinstance(v, (int, float))))
            raise ValueError(
                f"objective {obj.text!r}: unknown QOI {obj.qoi!r}.\n"
                f"Available for polarisation {obj.pol!r}: {available}")
        values[obj.column] = qois[obj.qoi]
    return values
