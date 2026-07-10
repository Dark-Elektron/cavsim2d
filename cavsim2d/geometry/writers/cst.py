"""Exporter that writes cavity cell parameters as a CST parameter file."""
import os
from cavsim2d.geometry.tangency import update_alpha


def write_cst_paramters(key, ic_, oc_l, oc_r, projectDir, cell_type, sub_dir='', opt=False, solver='NGSolveMEVP'):
    """
    Writes cavity geometric data that can be imported into CST Studio

    Parameters
    ----------
    key: str, int
        Cavity marker
    ic_: list, array like
        Inner cavity cell geometric variables
    oc_l: list, array like
        Outer cavity cell geometric variables
    projectDir: str
        Project directory
    cell_type: str
        Single cell or multicell

    Returns
    -------

    """
    ic_ = update_alpha(ic_)
    oc_l = update_alpha(oc_l)
    if solver.lower() == 'ngsolvemevp':
        folder = 'NGSolveMEVP'
    else:
        folder = 'Optimisation'

    if cell_type is None:
        path = os.path.join(projectDir, sub_dir, key, 'geometry', f'{key}.txt')

        with open(path, 'w') as f:
            name_list = ['Aeq', 'Beq', 'ai', 'bi', 'Ri', 'L', 'Req', 'alpha', 'Aeq_e', 'Beq_e', 'ai_e', 'bi_e', 'Ri_e',
                         'L_e', 'Req', 'alpha_e', 'key']

            value_list = [ic_[0], ic_[1], ic_[2], ic_[3], ic_[4], ic_[5], ic_[6], ic_[7],
                          oc_l[0], oc_l[1], oc_l[2], oc_l[3], oc_l[4], oc_l[5], oc_l[6], oc_l[7], key]

            for i in range(len(name_list)):
                if name_list[i] == 'key':
                    f.write(f'{name_list[i]} = "{0}" "{value_list[i]}"\n')
                else:
                    f.write(f'{name_list[i]} = "{value_list[i]}" ""\n')

    else:
        path = os.path.join(projectDir, sub_dir, key, 'geometry', f'{key}.txt')
        path_mc = os.path.join(projectDir, sub_dir, key, 'geometry', f'{key}_Multicell.txt')

        with open(path, 'w') as f:
            name_list = ['Aeq', 'Beq', 'ai', 'bi', 'Ri', 'L', 'Req', 'Aeq_e', 'Beq_e', 'ai_e', 'bi_e', 'Ri_e',
                         'L_e', 'Req_e', 'key']  # 'alpha_e', 'key']

            if cell_type == 'Mid Cell':
                value_list = [ic_[0], ic_[1], ic_[2], ic_[3], ic_[4], ic_[5], ic_[6],  #ic_[7],
                              'Aeq', 'Beq', 'ai', 'bi', 'Ri', 'L', 'Req', 'key']  #'alpha', key]
            else:
                value_list = [ic_[0], ic_[1], ic_[2], ic_[3], ic_[4], ic_[5], ic_[6],  #ic_[7],
                              oc_r[0], oc_r[1], oc_r[2], oc_r[3], oc_r[4], oc_r[5], ic_[6],  #oc_l[7],
                              oc_l[0], oc_l[1], oc_l[2], oc_l[3], oc_l[4], oc_l[5], ic_[6],  #oc_l[7],
                              key]

            for i in range(len(name_list)):
                if name_list[i] == 'key':
                    f.write(f'{name_list[i]} = "{0}" "{value_list[i]}"\n')
                else:
                    f.write(f'{name_list[i]} = "{value_list[i]}" ""\n')

        with open(path_mc, 'w') as f:
            name_list = ['Aeq', 'Beq', 'ai', 'bi', 'Ri', 'L', 'Req',  #'alpha',
                         'Aeq_er', 'Beq_er', 'ai_er', 'bi_er', 'Ri_er', 'L_er', 'Req',  #'alpha_er',
                         'Aeq_el', 'Beq_el', 'ai_el', 'bi_el', 'Ri_el', 'L_el', 'Req', 'key']  #'alpha_el', 'key']

            if cell_type == 'Mid Cell':
                value_list = [ic_[0], ic_[1], ic_[2], ic_[3], ic_[4], ic_[5], ic_[6],  #ic_[7],
                              'Aeq', 'Beq', 'ai', 'bi', 'Ri', 'L', 'Req',  #'alpha',
                              'Aeq', 'Beq', 'ai', 'bi', 'Ri', 'L', 'Req',  #'alpha',
                              key]
            else:
                value_list = [ic_[0], ic_[1], ic_[2], ic_[3], ic_[4], ic_[5], ic_[6],  #ic_[7],
                              oc_r[0], oc_r[1], oc_r[2], oc_r[3], oc_r[4], oc_r[5], ic_[6],  #oc_r[7],
                              oc_l[0], oc_l[1], oc_l[2], oc_l[3], oc_l[4], oc_l[5], ic_[6],  #oc_l[7],
                              key]

            for i in range(len(name_list)):
                if name_list[i] == 'key':
                    f.write(f'{name_list[i]} = "{0}" "{value_list[i]}"\n')
                else:
                    f.write(f'{name_list[i]} = "{value_list[i]}" ""\n')
