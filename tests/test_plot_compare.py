import os
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cavsim2d.study import Study

class DummyCavity:
    def __init__(self, name, plot_label, color, freq, e, b, k_cc, R_Q, G, GR_Q):
        self.name = name
        self.plot_label = plot_label
        self.color = color
        self.freq = freq
        self.e = e
        self.b = b
        self.k_cc = k_cc
        self.R_Q = R_Q
        self.G = G
        self.GR_Q = GR_Q

def test_plot_compare_fm_scatter_defaults(tmp_path):
    # Setup mock cavities
    cavs = Study(str(tmp_path), _skip_project_init=True)
    cav1 = DummyCavity('cav1', 'Cavity 1', 'blue', 800.0, 2.0, 4.0, 1.5, 110.0, 270.0, 29700.0)
    cav2 = DummyCavity('cav2', 'Cavity 2', 'red', 810.0, 2.1, 4.2, 1.6, 115.0, 275.0, 31625.0)
    
    cavs.cavities_list = [cav1, cav2]
    
    # Call plot_compare_fm_scatter (without uq)
    axd = cavs.eigenmode.plot_fm_scatter()
    
    # Verify the default axes are created
    expected_default_keys = ['freq [MHz]', 'Epk/Eacc []', 'Bpk/Eacc [mT/MV/m]', 'R/Q [Ohm]']
    assert set(axd.keys()) == set(expected_default_keys)
    plt.close('all')

def test_plot_compare_fm_scatter_custom_qois(tmp_path):
    cavs = Study(str(tmp_path), _skip_project_init=True)
    cav1 = DummyCavity('cav1', 'Cavity 1', 'blue', 800.0, 2.0, 4.0, 1.5, 110.0, 270.0, 29700.0)
    cav2 = DummyCavity('cav2', 'Cavity 2', 'red', 810.0, 2.1, 4.2, 1.6, 115.0, 275.0, 31625.0)
    
    cavs.cavities_list = [cav1, cav2]
    
    # Custom QoIs: frequency, R/Q, and kcc
    axd = cavs.eigenmode.plot_fm_scatter(qois=['frequency', 'r/q', 'kcc'])
    
    expected_keys = ['freq [MHz]', 'R/Q [Ohm]', 'kcc [%]']
    assert set(axd.keys()) == set(expected_keys)
    plt.close('all')

def test_plot_compare_fm_scatter_uq(tmp_path):
    cavs = Study(str(tmp_path), _skip_project_init=True)
    cav1 = DummyCavity('cav1', 'Cavity 1', 'blue', 800.0, 2.0, 4.0, 1.5, 110.0, 270.0, 29700.0)
    cav2 = DummyCavity('cav2', 'Cavity 2', 'red', 810.0, 2.1, 4.2, 1.6, 115.0, 275.0, 31625.0)
    
    cavs.cavities_list = [cav1, cav2]
    
    # Setup mock UQ and nominal results
    cavs.eigenmode_qois = {
        'Cavity 1': {
            'freq [MHz]': 800.0,
            'Epk/Eacc []': 2.0,
            'Bpk/Eacc [mT/MV/m]': 4.0,
            'R/Q [Ohm]': 110.0
        },
        'Cavity 2': {
            'freq [MHz]': 810.0,
            'Epk/Eacc []': 2.1,
            'Bpk/Eacc [mT/MV/m]': 4.2,
            'R/Q [Ohm]': 115.0
        }
    }
    
    cavs.uq_fm_results = {
        'Cavity 1': {
            'freq [MHz]': {'expe': [801.0], 'stdDev': [0.5]},
            'Epk/Eacc []': {'expe': [2.05], 'stdDev': [0.05]},
            'Bpk/Eacc [mT/MV/m]': {'expe': [4.05], 'stdDev': [0.1]},
            'R/Q [Ohm]': {'expe': [110.5], 'stdDev': [2.0]}
        },
        'Cavity 2': {
            'freq [MHz]': {'expe': [809.0], 'stdDev': [0.6]},
            'Epk/Eacc []': {'expe': [2.15], 'stdDev': [0.06]},
            'Bpk/Eacc [mT/MV/m]': {'expe': [4.25], 'stdDev': [0.12]},
            'R/Q [Ohm]': {'expe': [114.5], 'stdDev': [2.5]}
        }
    }
    
    # Custom list of UQ QoIs
    axd = cavs.eigenmode.plot_fm_scatter(uq=True, qois=['Epk', 'bpk'])
    
    expected_keys = ['Epk/Eacc []', 'Bpk/Eacc [mT/MV/m]']
    assert set(axd.keys()) == set(expected_keys)
    plt.close('all')
