import os
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.signal as sps
import numpy as np

from cavsim2d.constants import MROT_DICT, c0
from cavsim2d.utils.printing import info


class ABCIData:
    def __init__(self, dirc, fid, MROT):
        self.title_dict = {}
        self.data_dict = {}
        self.dir = Path(dirc)
        self.fid = fid
        self.MROT = MROT
        self.loss_factor = {}
        self.SIG = None
        self.wakelength = None
        self.checks(fid, MROT)

    def checks(self, fid, MROT):
        dirc = os.path.join(self.dir, fid, MROT_DICT[MROT], 'cavity.top')
        # try:
        self._get_plot_data(dirc)
        # except Exception as e:
        #     error(f"There seems to be a problem with the ABCI file directory. {e}")

    def _get_plot_data(self, dirc):
        frame_objects = {}
        frame_titles_objects = {}
        line_objects = []
        frame_title = []
        frame_count = 0

        plot_decorations = [r'Cavity Shape Input', r'Cavity Shape Used', r'Wake Potentials',
                            r'Real Part of Longitudinal Impedance', r'Imaginary Part of Longitudinal Impedance',
                            'Frequency Spectrum of Loss Factor', r'Loss Factor Spectrum Integrated upto F',
                            r'Real Part of Long. + Log Impedance', r'Imaginary Part of Long. + Log Impedance',
                            r'Spectrum of Long. + Log Loss Factor', r'Long. + Log Factor Integrated up to F',
                            r'Real Part of Azimuthal Impedance',
                            r'Imaginary Part of Azimuthal Impedance', r'Real Part of Transverse Impedance',
                            r'Imaginary Part of Transverse Impedance']

        with open(dirc, 'r') as f:
            for line in f:
                if 'NEW FRAME' in line:
                    # add frame to frame_objects
                    if frame_count > 0:
                        frame_titles_objects[frame_count - 1] = frame_title

                        # select suitable title
                        combined_title = ''.join(frame_title)
                        for decor in plot_decorations:
                            if decor in combined_title:
                                frame_objects[decor] = line_objects
                                break

                    # reset lists
                    line_objects = []
                    frame_title = []
                    new_frame = True

                    frame_count += 1

                if new_frame:
                    new_frame = False

                # add titles to frame_title
                if 'TITLE' in line or 'MORE' in line:
                    frame_title.append(line)
                    # if 'Transverse Wake' in line:
                    #     key = 'Transverse'

                    # get other parameters from title
                    # try:
                    if 'Azimuthal Wake' in line:
                        key = 'Azimuthal'
                    if 'Transverse Wake' in line:
                        key = 'Transverse'
                    if 'Longitudinal Wake' in line:
                        key = 'Longitudinal'

                    if 'Loss Factor' in line and key:
                        indx_0 = line.index('= ')
                        indx_1 = line.index('V/pC')
                        loss_factor = float(line[indx_0 + 2:indx_1])

                        self.loss_factor[key] = loss_factor
                        key = None

                    # get sigma
                    if 'SIG' in line and self.SIG is None:
                        indx_0 = line.index('SIG=')
                        indx_1 = line.index(' cm')
                        self.SIG = float(line[indx_0 + 5:indx_1])
                    # except:
                    #     pass

                if 'SET LIMITS' in line:
                    x, y = [], []

                    # get wakelength
                    if 'SET LIMITS X   0.00000E+00' in line:
                        indx_0 = line.index('SET LIMITS X   0.00000E+00')
                        indx_1 = line.index('Y')
                        self.wakelength = float(line[indx_0 + len('SET LIMITS X   0.00000E+00'):indx_1].strip())

                if 'JOIN' in line:
                    new_line_object = True
                else:
                    new_line_object = False

                if new_line_object:
                    # add x and y to line objects
                    line_objects.append([x, y])

                    # reset x and y
                    x, y = [], []

                if len(line.strip().split()) == 2 and line.strip().split()[0] != 'NEW' and line.strip().split()[
                    0] != 'JOIN':
                    new_line_object = False
                    ll = [float(a) for a in line.strip().split()]
                    x.append(ll[0])
                    y.append(ll[1])

            # EOF append last list
            frame_titles_objects[frame_count - 1] = frame_title

            # select suitable title
            combined_title = ''.join(frame_title)
            for decor in plot_decorations:
                if decor in combined_title:
                    frame_objects[decor] = line_objects
                    break

            self.data_dict = {}
            for key, lines in frame_objects.items():
                x, y = [], []
                for line in lines:
                    if len(line[0]) > 2:
                        x += line[0]
                        y += line[1]

                if key in [r'Cavity Shape Input', r'Cavity Shape Used']:
                    x = np.array(x) - np.amax(x)/2

                    # # insert values to complete shape
                    # x = np.insert(x, 0, x[0])
                    # y = np.insert(y, 0, 0)
                    # x = np.insert(x, -1, x[-1]).tolist()
                    # y = np.insert(y, -1, 0).tolist()

                self.data_dict[key] = [x, y]
        # include impedance magnitude
        if {'Real Part of Longitudinal Impedance', 'Imaginary Part of Longitudinal Impedance'}.issubset(self.data_dict.keys()):
            # longitudinal
            xr, yr = self.data_dict['Real Part of Longitudinal Impedance']
            xi, yi = self.data_dict['Imaginary Part of Longitudinal Impedance']
            y = (np.array(yr) ** 2 + np.array(yi) ** 2) ** 0.5
            xr = np.array(xr)
            self.data_dict['Longitudinal Impedance Magnitude'] = [xr, y]

        if {'Real Part of Transverse Impedance', 'Imaginary Part of Transverse Impedance'}.issubset(self.data_dict.keys()):
            # transverse
            xr, yr = self.data_dict['Real Part of Transverse Impedance']
            xi, yi = self.data_dict['Imaginary Part of Transverse Impedance']
            y = (np.array(yr) ** 2 + np.array(yi) ** 2) ** 0.5
            xr = np.array(xr)
            self.data_dict['Transversal Impedance Magnitude'] = [xr, y]

    def get_data(self, key):
        plot_decorations = [r'Cavity Shape Input', r'Cavity Shape Used', r'Wake Potentials',
                            r'Real Part of Longitudinal Impedance', r'Imaginary Part of Longitudinal Impedance',
                            'Frequency Spectrum of Loss Factor', r'Loss Factor Spectrum Integrated upto F',
                            r'Real Part of Long. + Log Impedance', r'Imaginary Part of Long. + Log Impedance',
                            r'Spectrum of Long. + Log Loss Factor', r'Long. + Log Factor Integrated up to F',
                            r'Real Part of Azimuthal Impedance',
                            r'Imaginary Part of Azimuthal Impedance', r'Real Part of Transverse Impedance',
                            r'Imaginary Part of Transverse Impedance']
        if isinstance(key, int):
            key = plot_decorations[key]

        self.x = self.data_dict[key][0]
        self.y = self.data_dict[key][1]

        # Process all needed data
        self.x_peaks, self.y_peaks = self.get_peaks(threshold=max(self.y) * 0.02)

        # if key != 'Loss Factor Spectrum Integrated up to F':
        #     self.bands = self._get_bands()
        #     self.fl_list = self._interp_fl()
        #     self.fr_list = self._interp_fr()
        #     self.BW = self._get_BW()
        #     self.Q = self._get_Q()
        #     self.R_Q = self._get_RQ()

        return self.x, self.y, self.fid

    def _nparray_to_list(self, x):
        result = []
        for array in x:
            result.extend(array.tolist())

        return result

    def get_peaks(self, threshold):
        # Develop later
        peaks, _ = sps.find_peaks(self.y, height=threshold)
        xp, yp = np.array(self.x)[peaks], np.array(self.y)[peaks]

        return xp, yp
        # self.plot_beam_pipe_cutoffs()

    def _get_bands(self):
        try:
            # get band ranges
            group_dict = {}
            gr_n = 0
            xl, yl = [], []
            n = 0
            for i, val in enumerate(self.x_peaks):
                if i == 0:
                    xl.append(val)
                    yl.append(self.y_peaks[i])
                else:
                    # if val - xl[n] < 0.2:
                    if val - xl[n] < 0.1:
                        xl.append(val)
                        yl.append(self.y_peaks[i])
                        n += 1
                    else:
                        # save group
                        group_dict[gr_n] = {'xb': xl, 'yb': yl}

                        # next group
                        gr_n += 1

                        # reset lists
                        xl, yl = [], []
                        n = 0

                        # append current result
                        xl.append(val)
                        yl.append(self.y[i])

                    group_dict[gr_n] = {'xb': xl, 'yb': yl}  # last group

            # print_(group_dict)
            return group_dict
        except Exception as e:
            info(f"Bands cannot be gotten without getting the peaks first. {e}")

    def _interp_fl(self):
        fl_list = []
        wx = [0, 0]
        wz = [1, 0]
        for f0, z in zip(self.x_peaks, self.y_peaks):
            z_0707 = 0.707 * z
            # print_(z_0707)

            l = 0
            for i, h in enumerate(self.y):
                if (l < z_0707 < h):

                    if (f0-0.01 < self.x[i] < f0):
                        wz = [l, h]
                        wx = [self.x[i - 1], self.x[i]]
                        break
                l = h

            # print_(f'\t{wx, wz}')

            # interpolate
            fl = wx[0] + (wx[1]-wx[0])*(z_0707-wz[0])/(wz[1]-wz[0])
            fl_list.append(fl)

        return fl_list

    def _interp_fr(self):
        fr_list = []
        wx = [0, 0]
        wz = [1, 0]
        for f0, z in zip(self.x_peaks, self.y_peaks):

            z_0707 = 0.707 * z
            # print_(z_0707)

            l = 0
            for i, h in enumerate(self.y):
                if l > z_0707 > h:
                    if (f0 < self.x[i] < f0 + 0.05):
                        wz = [l, h]
                        wx = [self.x[i - 1], self.x[i]]
                        break
                l = h

            # f0_old = f0
            # print_(f'\t{wx, wz}')

            # interpolate
            fr = wx[0] + (wx[1]-wx[0])*(z_0707-wz[0])/(wz[1]-wz[0])
            fr_list.append(fr)

        return fr_list

    def _get_Q(self):
        Q = [f0/bw for f0, bw in zip(self.x_peaks, self.BW)]
        return Q

    def _get_RQ(self):
        if self.MROT == 0:
            R_Q = [2*zz*1e3/q for zz, q in zip(self.y_peaks, self.Q)]
            # print_(f'R/Q: {R_Q}')
        else:
            c = c0
            k = [2*np.pi*f0*1e9/c for f0 in self.x_peaks]
            R_Q_m = [2*zz*1e3/q for zz, q in zip(self.y_peaks, self.Q)]
            R_Q = [r_q_m/kk for r_q_m, kk in zip(R_Q_m, k)]
            # print_(f'R/Q: {R_Q}')

        return R_Q

    def _get_BW(self):
        BW = [a - b for a, b in zip(self.fr_list, self.fl_list)]
        return BW

    def plot_bands(self):
        fig, axs = plt.subplots(2, sharex=True)
        fig.suptitle('Vertically stacked subplots')
        axs[1].bar(self.x_peaks, self.R_Q, width=0.01, alpha=1)


        axs[0].plot(self.x, self.y)
        axs[0].scatter(self.x_peaks, self.y_peaks)
        color = ['#008fd5', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b', '#810f7c', '#ed718b', '#2ccf8b', '#fded21', '#8248d2', '#4c3a27', '#000000',
                 '#008fd5', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b', '#810f7c', '#ed718b', '#2ccf8b', '#fded21', '#8248d2', '#4c3a27', '#000000']

        for key, val in self.bands.items():
            tol = 0.05
            lb = val['xb'][0] - tol
            rb = val['xb'][-1] + tol
            axs[0].axvspan(lb, rb, 0, 1, color=color[key], alpha=0.3)
            axs[1].axvspan(lb, rb, 0, 1, color=color[key], alpha=0.3)

        plt.show()
