from scipy.special import *
from cavsim2d.utils.shared_functions import *
from cavsim2d.constants import *
from scipy.special import jn_zeros, jnp_zeros

class QuickTools:
    def __init__(self):
        pass

    def make_bessel_mode_dict(self, max_m: int, max_n: int) -> dict[str, float]:
        """
        Return a dict whose keys are mode names like 'te21', 'tm03', ...
        up to m = 0..max_m and n = 1..max_n, and whose values are the
        corresponding Bessel‐zero (TE uses derivative‐zeros, TM uses function‐zeros).
        """
        mode_dict: dict[str, float] = {}
        for m in range(max_m + 1):
            # TM_mn : n-th zero of J_m
            tm_zeros = jn_zeros(m, max_n)
            for n, root in enumerate(tm_zeros, start=1):
                mode_dict[f"tm{m}{n}"] = root

            # TE_mn : n-th zero of J'_m
            te_zeros = jnp_zeros(m, max_n)
            for n, root in enumerate(te_zeros, start=1):
                mode_dict[f"te{m}{n}"] = root

        return self.sorted_by_value(mode_dict)

    @staticmethod
    def sorted_by_value(d: dict[str, float]) -> dict[str, float]:
        """Return a new dict with items sorted by ascending value."""
        return {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}

    def cwg_cutoff(self, r, l=0, mode=None):
        """

        Parameters
        ----------
        r: float, list
            radius or list of radii in nmm

        Returns
        -------
        f_cutoff: float, list
            cutoff frequency or list of cutoff frequencies in MHz
        """

        if isinstance(r, float) or isinstance(r, int):
            r = [r]
        if isinstance(l, float) or isinstance(l, int):
            l = [l]

        f_cutoff = {}
        mode_dict = {'te11': 1.841, 'tm01': 2.405, 'te21': 3.054, 'te01': 3.832, 'tm11': 3.832, 'tm21': 5.135,
                     'te12': 5.331, 'tm02': 5.520, 'te22': 6.706, 'te02': 7.016, 'tm12': 7.016, 'tm22': 8.417,
                     'te13': 8.536, 'tm03': 8.654, 'te23': 9.970, 'te03': 10.174, 'tm13': 10.174, 'tm23': 11.620}
        mode_dict = self.make_bessel_mode_dict(5, 5)
        print(mode_dict)

        if mode is None:
            for radius in r:
                f_cutoff[f'{radius}'] = {}
                for mode, j in mode_dict.items():
                    f = c0 / (2 * np.pi) * (j / (radius * 1e-3))
                    f_cutoff[f'{radius}'][mode] = f * 1e-6
        else:
            if isinstance(mode, list):
                for mode_ in mode:
                    for radius in r:
                        f_cutoff[f'{radius}'] = {}
                        try:
                            j = mode_dict[mode_]
                        except KeyError:
                            error("One or more mode names is wrong. Please check mode names.")
                            j = 0
                        f = c0 / (2 * np.pi) * (j / (radius * 1e-3))
                        f_cutoff[f'{radius}'][mode_] = f * 1e-6
            else:
                if isinstance(mode, str):
                    for radius in r:
                        f_cutoff[f'{radius}'] = {}
                        try:
                            j = mode_dict[mode]
                        except KeyError:
                            error("One or more mode names is wrong. Please check mode names.")
                            j = 0
                        f = c0 / (2 * np.pi) * (j / (radius * 1e-3))
                        f_cutoff[f'{radius}'][mode] = f * 1e-6
                else:
                    error("One or more mode names is wrong. Please check mode names.")

        return f_cutoff

    @staticmethod
    def rwg_cutoff(a, b, mn=None, l=0, p=None):
        if isinstance(a, float) or isinstance(a, int):
            a = [a]
        if isinstance(b, float) or isinstance(b, int):
            b = [b]
        if isinstance(l, float) or isinstance(l, int):
            l = [l]

        if mn is None:
            mn = [[0, 1]]

        if len(np.array(mn).shape) == 1:
            mn = [mn]

        if isinstance(p, int):
            p = [p]

        if p is None:
            p = 0

        f_cutoff = {}
        try:
            for a_ in a:
                f_cutoff[f'a: {a_} mm'] = {}
                for b_ in b:
                    f_cutoff[f'a: {a_} mm'][f'b: {b_} mm'] = {}
                    for l_ in l:
                        f_cutoff[f'a: {a_} mm'][f'b: {b_} mm'] = {}
                        for mn_ in mn:
                            m, n = mn_
                            if l_ == 0:
                                f = (c0 / (2 * np.pi)) * (
                                        (m * np.pi / (a_ * 1e-3)) ** 2 + (n * np.pi / (b_ * 1e-3)) ** 2) ** 0.5
                                f_cutoff[f'a: {a_} mm'][f'b: {b_} mm'][f'TE/TM({m},{n})'] = f * 1e-6
                            else:
                                f_cutoff[f'a: {a_} mm'][f'b: {b_} mm'][f'l: {l_} mm'] = {}
                                for p_ in p:
                                    f = (c0 / (2 * np.pi)) * (
                                            (m * np.pi / (a_ * 1e-3)) ** 2 + (n * np.pi / (b_ * 1e-3)) ** 2 + (
                                            p_ * np.pi / (l_ * 1e-3)) ** 2) ** 0.5
                                    f_cutoff[f'a: {a_} mm'][f'b: {b_} mm'][f'l: {l_} mm'][
                                        f'TE/TM({m},{n},{p_})'] = f * 1e-6
            return f_cutoff
        except ValueError:
            print("Please enter a valid number.")

    @staticmethod
    def coaxial_tline(D, d, x=0, epsr=1):
        D, d, x = D * 1e-3, d * 1e-3, x * 1e-3

        Z = 60 * np.arccosh((d ** 2 + D ** 2 - (2*x) ** 2) / (2 * d * D))
        C = 2 * np.pi * eps0 * epsr / (np.arccosh((d ** 2 + D ** 2 - (2*x) ** 2) / (2 * d * D))) * 1e12
        L = mu0 / (2 * np.pi) * np.arccosh((d ** 2 + D ** 2 - (2*x) ** 2) / (2 * d * D)) * 1e9
        return {"L' [nH/m]": L, "C' [pF/m]": C, "Z' [Ohm/m]": Z}

    @staticmethod
    def parallel_plate_capacitor(l, b, d, epsr=1):
        l, b = l * 1e-3, b * 1e-3
        C = epsr * l * b / d
        return {"C' [pF]": C}

    @staticmethod
    def parallel_disc_capacitor(D, d, epsr=1):
        D, d = D * 1e-3, d * 1e-3
        C = epsr * np.pi * D ** 2 / (4 * d)
        return {"C' [pF]": C}

    @staticmethod
    def cwg_analytical(m, n, kind='te', R=None, pol=None, component='abs'):
        if R is None:
            R = 1
        if not pol:
            pol = 0
        else:
            pol = 0

        r_ = np.linspace(1e-6, R, 500)
        t_ = np.linspace(0, 2 * np.pi, 500)
        radius, theta = np.meshgrid(r_, t_)
        A = 1
        k = 0  # no propagation in z

        if kind.lower() == 'te':
            j_mn_p = jnp_zeros(m, n)[n - 1]
            kc = j_mn_p / R
            w = kc / np.sqrt(mu0 * eps0)

            beta = np.sqrt(k ** 2 - kc ** 2)

            Er = -1j * w * mu0 * m / (kc ** 2 * radius) * A * (np.cos(m * theta + pol) - np.sin(m * theta + pol)) * jv(
                m,
                kc * radius)
            Et = 1j * w * mu0 / kc * A * (np.sin(m * theta + pol) + np.cos(m * theta + pol)) * jvp(m, kc * radius)
            Ez = 0
            Hr = -1j * beta / kc * A * (np.sin(m * theta + pol) + np.cos(m * theta + pol)) * jvp(m, kc * radius)
            Ht = -1j * beta * m / (kc ** 2 * radius) * A * (np.cos(m * theta + pol) - np.sin(m * theta + pol)) * jv(m,
                                                                                                                    kc * radius)
            Hz = A * (np.sin(m * theta + pol) + np.cos(m * theta + pol)) * jv(m, kc * radius)

            Emag = np.abs(np.sqrt(Er ** 2 + Et ** 2 + Ez ** 2))
            Hmag = np.abs(np.sqrt(Hr ** 2 + Ht ** 2 + Hz ** 2))
        elif kind.lower() == 'tm':
            j_mn = jn_zeros(m, n)[n - 1]
            kc = j_mn / R
            w = kc / np.sqrt(mu0 * eps0)
            beta = np.sqrt(k ** 2 - kc ** 2)
            Ez = A * (np.sin(m * theta + pol) + np.cos(m * theta + pol)) * jv(m, kc * radius)
            Er = -1j * beta / kc * A * (np.sin(m * theta + pol) + np.cos(m * theta + pol)) * jvp(m, kc * radius)
            Et = -1j * beta * m / (kc ** 2 * radius) * A * (np.cos(m * theta + pol) - np.sin(m * theta + pol)) * jv(m,
                                                                                                                    kc * radius)
            Hr = 1j * w * eps0 * m / (kc ** 2 * radius) * A * (np.cos(m * theta + pol) - np.sin(m * theta + pol)) * jv(
                m, kc * radius)
            Ht = -1j * w * eps0 / kc * A * (np.sin(m * theta + pol) + np.cos(m * theta + pol)) * jvp(m, kc * radius)
            Hz = 0

            Emag = np.abs(np.sqrt(Er ** 2 + Et ** 2 + Ez ** 2))
            Hmag = np.abs(np.sqrt(Hr ** 2 + Ht ** 2 + Hz ** 2))
        else:
            raise RuntimeError('Please enter valid cicular waveguide mode kind.')

        if component.lower() == 'abs':
            return radius, theta, Emag, Hmag
        if component.lower() == 'azimuthal':
            return radius, theta, Et, Ht
        if component.lower() == 'radial':
            return radius, theta, Er, Hr
        if component.lower() == 'longitudinal':
            return radius, theta, Ez, Hz

    @staticmethod
    def cwg_tm_analytical(m, n, theta, radius, R=None, pol=None, component='abs'):
        if R is None:
            R = 1
        if pol is None:
            pol = 0

        r_ = np.linspace(1e-6, R, 500)
        t_ = np.linspace(0, 2 * np.pi, 500)
        radius, theta = np.meshgrid(r_, t_)

        j_mn = jn_zeros(m, n)[n - 1]
        A = 1
        k = 0  # no propagation in z
        kc = j_mn / R
        w = kc / np.sqrt(mu0 * eps0)
        beta = np.sqrt(k ** 2 - kc ** 2)
        Ez = A * (np.sin(m * theta + pol) + np.cos(m * theta + pol)) * jv(m, kc * radius)
        Er = -1j * beta / kc * A * (np.sin(m * theta + pol) + np.cos(m * theta + pol)) * jvp(m, kc * radius)
        Et = -1j * beta * m / (kc ** 2 * radius) * A * (np.cos(m * theta + pol) - np.sin(m * theta + pol)) * jv(m,
                                                                                                                kc * radius)
        Hr = 1j * w * eps0 * m / (kc ** 2 * radius) * A * (np.cos(m * theta + pol) - np.sin(m * theta + pol)) * jv(m,
                                                                                                                   kc * radius)
        Ht = -1j * w * eps0 / kc * A * (np.sin(m * theta + pol) + np.cos(m * theta + pol)) * jvp(m, kc * radius)
        Hz = 0

        Emag = np.abs(np.sqrt(Er ** 2 + Et ** 2 + Ez ** 2))
        Hmag = np.abs(np.sqrt(Hr ** 2 + Ht ** 2 + Hz ** 2))

        if component.lower() == 'abs':
            return radius, theta, Emag, Hmag
        if component.lower() == 'azimuthal':
            return radius, theta, Et, Ht
        if component.lower() == 'radial':
            return radius, theta, Er, Hr
        if component.lower() == 'longitudinal':
            return radius, theta, Ez, Hz