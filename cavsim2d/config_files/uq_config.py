uq_config = {
    'option': True,
    'variables': ['A', 'B'],
    # 'objectives': ["freq [MHz]", "R/Q [Ohm]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "G [Ohm]", "kcc [%]", "ff [%]"],
    'objectives': ["Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]", "G [Ohm]"],
    # 'objectives': ["ZL"],
    'delta': [0.03, 0.03],
    'processes': 3,
    'distribution': 'gaussian',
    # 'method': ['QMC', 'LHS', 1000],
    # 'method': ['QMC', 'Sobol', 1000],
    # 'method': ['Qudrature', 'Gaussian', 1000],
    'method': ['Quadrature', 'Stroud3'],
    # 'method': ['Quadrature', 'Stroud5'],
    # 'gaussian': ['Quadrature', 'Gaussian'],
    # 'from file': ['<file path>', columns],
    'cell type': 'mid-cell',
    'cell_complexity': 'multicell'
}