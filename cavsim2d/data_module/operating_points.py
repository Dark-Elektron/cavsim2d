import os
import json
import re
import numpy as np
import pandas as pd

#: Operating-point key holding the RMS bunch length(s), in mm. ``'sigma'`` is
#: accepted as a short alias.
SIGMA_KEY = 'sigma [mm]'

# The original spelling: one key per named bunch length, 'sigma_SR [mm]' and
# 'sigma_BS [mm]'. Still read, so saved configs and the built-in tables keep working.
_LEGACY_SIGMA = re.compile(r'^sigma_(\w+)\s*\[mm\]$')


def bunch_lengths(op_point):
    """The bunch lengths an operating point asks for, as ``{label: sigma [mm]}``.

    One bunch length::

        {'I0 [mA]': 1280, 'Nb [1e11]': 2.76, 'sigma [mm]': 4.32}         # {'': 4.32}

    Several, each under a label of your choice (for instance the
    synchrotron-radiation and beamstrahlung lengths of a collider)::

        {..., 'sigma [mm]': {'SR': 4.32, 'BS': 15.2}}                     # {'SR': 4.32, 'BS': 15.2}

    The older ``'sigma_SR [mm]'``/``'sigma_BS [mm]'`` keys are still read. The
    order is preserved, and the first entry is the operating point's primary bunch
    length, the one the comparison plots and tables show.
    """
    out = {}
    value = op_point.get(SIGMA_KEY, op_point.get('sigma'))
    if isinstance(value, dict):
        out.update({str(k): float(v) for k, v in value.items()})
    elif value is not None:
        out[''] = float(value)
    for key, v in op_point.items():
        match = _LEGACY_SIGMA.match(str(key))
        if match and match.group(1) not in out:
            out[match.group(1)] = float(v)
    if not out:
        raise KeyError(
            f"operating point has no bunch length: give '{SIGMA_KEY}' as one value "
            f"(e.g. 4.32) or as named values (e.g. {{'SR': 4.32, 'BS': 15.2}}).")
    return out


def bunch_tag(op_name, label, sigma_mm):
    """The id a bunch-length sub-run is filed under: ``'Z_SR_4.32mm'``, or
    ``'Z_4.32mm'`` for an operating point with a single, unlabelled bunch length.
    Every reader and writer builds it here, so the two cannot disagree on
    formatting."""
    sigma_mm = float(sigma_mm)
    return f'{op_name}_{label}_{sigma_mm}mm' if label else f'{op_name}_{sigma_mm}mm'


class OperationPoints:
    def __init__(self, filepath=None):
        self.op_points = {}

        if filepath:
            if os.path.exists(filepath):
                self.op_points = self.load_operating_point(filepath)

    def load_operating_point(self, filepath):
        with open(filepath, 'r') as f:
            op_points = json.load(f)

        self.op_points = op_points
        return op_points

    def get_default_operating_points(self):
        self.op_points = pd.DataFrame({
            "Z_2023": {
                "freq [MHz]": 400.79,
                "E [GeV]": 45.6,
                "I0 [mA]": 1280,
                "V [GV]": 0.12,
                "Eacc [MV/m]": 5.72,
                "nu_s []": 0.0370,
                "alpha_p [1e-5]": 2.85,
                "tau_z [ms]": 354.91,
                "tau_xy [ms]": 709.82,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 56,
                "N_c []": 56,
                "T [K]": 4.5,
                "sigma_SR [mm]": 4.32,
                "sigma_BS [mm]": 15.2,
                "Nb [1e11]": 2.76
            },
            "W_2023": {
                "freq [MHz]": 400.79,
                "E [GeV]": 80,
                "I0 [mA]": 135,
                "V [GV]": 1.0,
                "Eacc [MV/m]": 10.61,
                "nu_s []": 0.0801,
                "alpha_p [1e-5]": 2.85,
                "tau_z [ms]": 65.99,
                "tau_xy [ms]": 131.98,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 132,
                "T [K]": 4.5,
                "sigma_SR [mm]": 3.55,
                "sigma_BS [mm]": 7.02,
                "Nb [1e11]": 2.29
            },
            "H_2023": {
                "freq [MHz]": 400.79,
                "E [GeV]": 120,
                "I0 [mA]": 53.4,
                "V [GV]": 2.1,
                "Eacc [MV/m]": 10.61,
                "nu_s []": 0.0328,
                "alpha_p [1e-5]": 0.733,
                "tau_z [ms]": 19.6,
                "tau_xy [ms]": 39.2,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 528,
                "T [K]": 4.5,
                "sigma_SR [mm]": 2.5,
                "sigma_BS [mm]": 4.45,
                "Nb [1e11]": 1.51
            },
            "ttbar_2023": {
                "freq [MHz]": 801.58,
                "E [GeV]": 182.5,
                "I0 [mA]": 10,
                "V [GV]": 9.2,
                "Eacc [MV/m]": 20.12,
                "nu_s []": 0.0826,
                "alpha_p [1e-5]": 0.733,
                "tau_z [ms]": 5.63,
                "tau_xy [ms]": 11.26,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 488,
                "T [K]": 2,
                "sigma_SR [mm]": 1.67,
                "sigma_BS [mm]": 2.54,
                "Nb [1e11]": 2.26
            },
            "Z_2022": {
                "freq [MHz]": 400.79,
                "E [GeV]": 45.6,
                "I0 [mA]": 1400,
                "V [GV]": 0.12,
                "Eacc [MV/m]": 5.72,
                "nu_s []": 0.0370,
                "alpha_p [1e-5]": 2.85,
                "tau_z [ms]": 354.91,
                "tau_xy [ms]": 709.82,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 56,
                "T [K]": 4.5,
                "sigma_SR [mm]": 4.32,
                "sigma_BS [mm]": 15.2,
                "Nb [1e11]": 2.76
            },
            "W_2022": {
                "freq [MHz]": 400.79,
                "E [GeV]": 80,
                "I0 [mA]": 135,
                "V [GV]": 1.0,
                "Eacc [MV/m]": 11.91,
                "nu_s []": 0.0801,
                "alpha_p [1e-5]": 2.85,
                "tau_z [ms]": 65.99,
                "tau_xy [ms]": 131.98,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 112,
                "T [K]": 4.5,
                "sigma_SR [mm]": 3.55,
                "sigma_BS [mm]": 7.02,
                "Nb [1e11]": 2.29
            },
            "H_2022": {
                "freq [MHz]": 400.79,
                "E [GeV]": 120,
                "I0 [mA]": 53.4,
                "V [GV]": 2.1,
                "Eacc [MV/m]": 10.61,
                "nu_s []": 0.0328,
                "alpha_p [1e-5]": 0.733,
                "tau_z [ms]": 19.6,
                "tau_xy [ms]": 39.2,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 528,
                "T [K]": 4.5,
                "sigma_SR [mm]": 2.5,
                "sigma_BS [mm]": 4.45,
                "Nb [1e11]": 1.51
            },
            "ttbar_2022": {
                "freq [MHz]": 801.58,
                "E [GeV]": 182.5,
                "I0 [mA]": 10,
                "V [GV]": 9.2,
                "Eacc [MV/m]": 20.12,
                "nu_s []": 0.0826,
                "alpha_p [1e-5]": 0.733,
                "tau_z [ms]": 5.63,
                "tau_xy [ms]": 11.26,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 488,
                "T [K]": 2,
                "sigma_SR [mm]": 1.67,
                "sigma_BS [mm]": 2.54,
                "Nb [1e11]": 2.26
            },
            "Z_booster_2022": {
                "freq [MHz]": 801.58,
                "E [GeV]": 45.6,
                "I0 [mA]": 128,
                "V [GV]": 0.14,
                "Eacc [MV/m]": 6.23,
                "nu_s []": 0.0370,
                "alpha_p [1e-5]": 2.85,
                "tau_z [ms]": 354.91,
                "tau_xy [ms]": 709.82,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 120,
                "T [K]": 4.5,
                "sigma_SR [mm]": 4.32,
                "sigma_BS [mm]": 15.2,
                "Nb [1e11]": 0.276
            },
            "Z_2018": {
                "freq [MHz]": 400.79,
                "E [GeV]": 45.6,
                "I0 [mA]": 1390,
                "V [GV]": 0.10,
                "Eacc [MV/m]": 5.72,
                "nu_s []": 0.025,
                "alpha_p [1e-5]": 1.48,
                "tau_z [ms]": 424.6,
                "tau_xy [ms]": 849.2,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 52,
                "T [K]": 4.5,
                "sigma_SR [mm]": 3.5,
                "sigma_BS [mm]": 12.1,
                "Nb [1e11]": 1.7
            },
            "W_2018": {
                "freq [MHz]": 400.79,
                "E [GeV]": 80,
                "I0 [mA]": 147,
                "V [GV]": 0.75,
                "Eacc [MV/m]": 11.91,
                "nu_s []": 0.0506,
                "alpha_p [1e-5]": 1.48,
                "tau_z [ms]": 78.7,
                "tau_xy [ms]": 157.4,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 52,
                "T [K]": 4.5,
                "sigma_SR [mm]": 3.0,
                "sigma_BS [mm]": 6.0,
                "Nb [1e11]": 1.5
            },
            "H_2018": {
                "freq [MHz]": 400.79,
                "E [GeV]": 120,
                "I0 [mA]": 29,
                "V [GV]": 2.0,
                "Eacc [MV/m]": 11.87,
                "nu_s []": 0.036,
                "alpha_p [1e-5]": 0.73,
                "tau_z [ms]": 23.4,
                "tau_xy [ms]": 46.8,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 136,
                "T [K]": 4.5,
                "sigma_SR [mm]": 3.15,
                "sigma_BS [mm]": 5.3,
                "Nb [1e11]": 1.8
            },
            "ttbar_2018": {
                "freq [MHz]": 801.58,
                "E [GeV]": 182.5,
                "I0 [mA]": 10.8,
                "V [GV]": 10.93,
                "Eacc [MV/m]": 24.72,
                "nu_s []": 0.087,
                "alpha_p [1e-5]": 0.73,
                "tau_z [ms]": 6.8,
                "tau_xy [ms]": 13.6,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 584,
                "T [K]": 2,
                "sigma_SR [mm]": 1.97,
                "sigma_BS [mm]": 2.54,
                "Nb [1e11]": 2.3
            }
        })


