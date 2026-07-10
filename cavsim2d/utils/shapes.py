"""Shape-dictionary manipulation: continuity, perturbation and tabulation."""
import copy
import json
import os
import re
import numpy as np
import pandas as pd
from cavsim2d.utils.data_utils import VAR_NAMES
from cavsim2d.utils.quadrature import generate_nodes


def enforce_Req_continuity(par_mid, par_end_l, par_end_r, cell_type=None):
    """
    Enforce continuity at iris and equator of cavities

    Parameters
    ----------
    par_mid
    par_end_l
    par_end_r
    cell_type

    Returns
    -------

    """

    if cell_type:
        ct = cell_type.lower().replace('-', ' ').replace('_', ' ')
        if ct == 'mid cell':
            par_mid[6] = par_end_r[6]
            par_end_l[6] = par_end_r[6]
        elif ct == 'end cell':
            par_end_l[6] = par_mid[6]
            par_end_r[6] = par_mid[6]
        elif ct == 'single cell':
            par_mid[6] = par_end_r[6]
            par_end_l[6] = par_end_r[6]
        else:
            par_mid[6] = par_end_r[6]
            par_end_l[6] = par_end_r[6]
    else:
        Req_avg = (par_mid[6] + par_end_l[6] + par_end_r[6]) / 3
        par_mid[6] = Req_avg
        par_end_l[6] = Req_avg
        par_end_r[6] = Req_avg


def save_tune_result(d, folder, filename):
    with open(os.path.join(folder, 'eigenmode', filename), 'w') as file:
        file.write(json.dumps(d, indent=4, separators=(',', ': ')))


def to_multicell(n_cells, shape):
    shape_multicell = {}
    mid_cell = shape['IC']
    mid_cell_multi = np.array([[[a, a] for _ in range(n_cells - 1)] for a in mid_cell])

    shape_multicell['OC'] = shape['OC']
    shape_multicell['OC_R'] = shape['OC_R']
    shape_multicell['IC'] = mid_cell_multi
    # shape_multicell['BP'] = shape['BP']
    shape_multicell['n_cells'] = shape['n_cells']
    shape_multicell['CELL TYPE'] = 'multicell'

    return shape_multicell




def expand_cells(cav: dict, cells):
    """
    Given shape['n_cells'], expand 'all' to every half-cell:
      ['cell1_l','cell1_r',...,'cellN_l','cellN_r']
    Or accept a single string or list.
    """
    N = cav.n_cells
    if isinstance(cells, str) and cells == 'all':
        return [f'cell{i}_{side}' for i in range(1, N + 1) for side in ('l', 'r')]
    if isinstance(cells, str):
        return [cells]
    return list(cells)


def apply_perturbation(base,
                       deltas: list,
                       perturbed_vars: list,
                       mode: str):
    """
    mode='add': x_new = x + δ
    mode='mul': x_new = x * (1 + δ)
    """
    P     = len(VAR_NAMES)
    N     = base.n_cells
    out   = {}
    for ii, delta in enumerate(deltas):
        cav    = copy.deepcopy(base)
        # one slot for the very left half-cell

        # choose apply function. Go through the model's accessors, not
        # cav.parameters[...], so a variable that is not a plain scalar slot
        # (a spline's 'p3_r' lives inside the control point [z, r]) perturbs too.
        for pvar, d in zip(perturbed_vars, delta):
            x = cav.get_tune_value(pvar)
            cav.set_tune_value(pvar, x + d if mode == 'add' else x * (1 + d))

        # rename
        new_name = f'{cav.name}_Q{ii}'
        cav.name = new_name
        cav.projectDir = cav.uq_dir
        out[new_name] = cav
    return out


HALF_CELL_COLS = ('A', 'B', 'a', 'b', 'Ri', 'L', 'Req')


def half_cell_free_variables(n_cells, variables):
    """The *free* random variables of a multicell cavity, honouring continuity.

    A half-cell array has ``2 * n_cells`` rows, but its entries are not all
    independent:

    - ``Req`` is shared by the two halves of a cell  -> one variable per cell
    - ``Ri`` is shared across an iris plane          -> one variable per plane,
      i.e. ``n_cells + 1`` of them (two apertures and ``n_cells - 1`` irises)
    - everything else is per half-cell               -> ``2 * n_cells`` variables

    Returns a list of ``(label, column, rows)``: perturbing a variable adds the
    same delta to every row it spans, so the constraints hold by construction.
    """
    spec = []
    for var in variables:
        if var not in HALF_CELL_COLS:
            raise ValueError(f'unknown half-cell variable {var!r}; '
                             f'expected one of {HALF_CELL_COLS}')
        col = HALF_CELL_COLS.index(var)
        if var == 'Req':
            for k in range(n_cells):
                spec.append((f'Req_cell{k + 1}', col, [2 * k, 2 * k + 1]))
        elif var == 'Ri':
            spec.append(('Ri_aperture_l', col, [0]))
            for k in range(1, n_cells):
                spec.append((f'Ri_iris{k}', col, [2 * k - 1, 2 * k]))
            spec.append(('Ri_aperture_r', col, [2 * n_cells - 1]))
        else:
            for row in range(2 * n_cells):
                side = 'l' if row % 2 == 0 else 'r'
                spec.append((f'{var}_cell{row // 2 + 1}_{side}', col, [row]))
    return spec


def perturb_half_cells(cav, config):
    """Perturb a multicell cavity's half-cells, one random variable per free entry.

    Returns ``({name: half_cell_array}, weights)``. Continuity is preserved by
    construction: shared entries (a cell's equator, an iris plane) move together,
    so no post-hoc projection is needed.
    """
    uq_config = config['uq_config']
    variables = uq_config['variables']
    if isinstance(variables, str):
        variables = [variables]

    if 'perturbation_mode' not in uq_config:
        uq_config['perturbation_mode'] = ['add', uq_config.get('delta', 0.01)]
    mode = uq_config['perturbation_mode']
    if len(mode) < 2:
        mode.append(uq_config.get('delta', 0.01))

    deltas_per_var = mode[1]
    if not isinstance(deltas_per_var, (list, tuple, np.ndarray)):
        deltas_per_var = [deltas_per_var] * len(variables)

    base = cav.half_cells()
    spec = half_cell_free_variables(cav.n_cells, variables)

    # a free variable inherits the bound of the geometry variable it came from
    bounds = []
    for var, bound in zip(variables, deltas_per_var):
        bounds.extend([bound] * sum(1 for s in spec if s[1] == HALF_CELL_COLS.index(var)))

    nodes, weights = generate_nodes(len(spec), bounds, uq_config['method'])

    out = {}
    for i, node in enumerate(nodes):
        hc = np.array(base, dtype=float)
        for (label, col, rows), d in zip(spec, node):
            for row in rows:
                if mode[0] == 'add':
                    hc[row, col] += d
                else:
                    hc[row, col] *= (1 + d)
        out[f'{cav.name}_Q{i}'] = hc
    return out, np.atleast_2d(weights).T


def half_cells_to_dataframe(perturbed):
    """Half-cell arrays -> a node table with columns ``A1, B1, ..., Req1, A2, ...``,
    one index per half-cell (left then right, across all cells)."""
    rows = []
    for name, hc in perturbed.items():
        hc = np.asarray(hc, dtype=float)
        row = {'name': name}
        for i, half in enumerate(hc, start=1):
            for col, var in enumerate(HALF_CELL_COLS):
                row[f'{var}{i}'] = half[col]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.set_index('name', inplace=True)
    return df


def generate_perturbed_shapes(shape: dict,
                              cells,
                              variables: list,
                              mode: list,
                              node_type: list):
    """
    High-level API: returns (shapes, weights).

    - cells: 'all' or list of 'cellX_l'/'cellX_r'
    - variables: subset of VAR_NAMES
    - bound: absolute delta bound
    - mode: 'add' or 'mul'
    - n: nodes per dimension (ignored by stroud3)
    - node_type: 'uniform','gauss_legendre','stroud3'
    """
    cell_list = expand_cells(shape, cells)
    cell_vars = [(c, v) for c in cell_list for v in variables]
    k = len(cell_vars)

    deltas, weights = generate_nodes(k, mode[1], node_type)

    shapes = apply_perturbation(shape, deltas, cell_vars, mode[0])

    return shapes, np.atleast_2d(weights).T


def perturb_geometry(cav, eigenmode_config):

    uq_config = eigenmode_config['uq_config']
    uq_vars = uq_config['variables']
    # which_cell = uq_config['cell']

    method = uq_config['method']
    if 'perturbation_mode' not in uq_config:
        # default: additive perturbation with bound from delta or 0.01
        uq_config['perturbation_mode'] = ['add', uq_config.get('delta', 0.01)]

    perturbation_mode = uq_config['perturbation_mode']
    if len(perturbation_mode) < 2:
        perturbation_mode.append(uq_config.get('delta', 0.01))

    if not isinstance(perturbation_mode[1], list):
        perturbation_mode[1] = [perturbation_mode[1]] * len(uq_vars)

    # cells = which_cell
    variables = uq_vars
    mode = perturbation_mode
    node_type = method

    # get perturbed variables
    # get perturbed variables
    uq_parameters = uq_config['variables']
    if isinstance(uq_parameters, str):
        uq_parameters = [uq_parameters]

    # Map deltas to expanded variables
    config_deltas = uq_config.get('delta', 0.01)
    if not isinstance(config_deltas, (list, np.ndarray, list)):
        config_deltas = [config_deltas] * len(uq_parameters)
        
    if uq_config.get('cell', 'all') == 'all':
        # Ask the model which parameter slots each variable names. The old code
        # substring-matched the variable against parameter keys, which found
        # nothing for a spline's 'p3_r' (k = 0 -> ZeroDivisionError in the
        # quadrature) and matched the pillbox's 'L_bp' when asked for 'L'.
        perturbed_vars = []
        full_delta = []
        for i, k_var in enumerate(uq_parameters):
            for par in cav.expand_variable(k_var):
                perturbed_vars.append(par)
                full_delta.append(config_deltas[i])
    else:
        perturbed_vars = uq_parameters
        full_delta = config_deltas

    k = len(perturbed_vars)
    mode[1] = full_delta

    deltas, weights = generate_nodes(k, mode[1], node_type)
    perturbed_cavs_dict = apply_perturbation(cav, deltas, perturbed_vars, mode[0])

    return perturbed_cavs_dict, np.atleast_2d(weights).T


def enforce_continuity_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce:
      Req1==Req2, Req3==Req4, Req5==Req6, ...
      Ri2==Ri3, Ri4==Ri5, Ri6==Ri7, ...
    in a DataFrame with columns 'Req1'...'ReqN' and 'Ri1'...'RiN'.
    """
    df2 = df.copy()
    pat = re.compile(r'^(Req|Ri)(\d+|_[a-zA-Z0-9]+)$')

    # collect column names by var and index
    req = {}  # idx -> col
    ri  = {}
    for col in df2.columns:
        m = pat.match(col)
        if not m: continue
        var, idx = m.group(1), int(m.group(2))
        (req if 'Req' in var else ri)[idx] = col

    max_idx = max(req.keys() | ri.keys())

    # Equator: Req at odd indices paired with next
    for i in range(1, max_idx+1, 2):
        c1 = req.get(i)
        c2 = req.get(i+1)
        if c1 and c2:
            avg = 0.5*(df2[c1] + df2[c2])
            df2[c1] = avg
            df2[c2] = avg

    # Iris: Ri at even indices paired with next
    for i in range(2, max_idx+1, 2):
        c1 = ri.get(i)
        c2 = ri.get(i+1)
        if c1 and c2:
            avg = 0.5*(df2[c1] + df2[c2])
            df2[c1] = avg
            df2[c2] = avg

    return df2


def shapes_to_dataframe(cavs_dict):
    """
    Convert a list of perturbed-shape dicts into a DataFrame.

    Columns are named A1, B1, a1, ..., A2, B2, a2, ... etc.,
    where each half-cell (left then right) across all cells
    is assigned an increasing index.
    """
    if not cavs_dict:
        return pd.DataFrame()

    # Build a list of rows from each object's `.parameter` dict
    data = []
    for name, cav in cavs_dict.items():
        row = cav.parameters.copy()  # extract the parameter dictionary
        row['name'] = name  # optionally include the cavity name as a column
        data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Optionally set the name as index
    df.set_index('name', inplace=True)

    return df


def make_dirs_from_dict(d, current_dir):
    for key, val in d.items():
        if not os.path.exists(os.path.join(current_dir, key)):
            os.mkdir(os.path.join(current_dir, key))
            if type(val) == dict:
                make_dirs_from_dict(val, os.path.join(current_dir, key))
        elif val:
            make_dirs_from_dict(val, os.path.join(current_dir, key))
