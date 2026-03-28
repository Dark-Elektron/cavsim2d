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

    def cwg_cutoff(self, r, l=0, mode=None, epsr=1, mu0r=1, p_max=3):

        if isinstance(r, (float, int)):
            r = [r]
        if isinstance(l, (float, int)):
            l = [l]

        mode_dict = self.make_bessel_mode_dict(5, 5)

        # --------------------------------------------------
        # Mode selection
        # --------------------------------------------------
        if mode is None:
            selected_modes = mode_dict
        else:
            if isinstance(mode, str):
                mode = [mode]

            selected_modes = {}
            for m in mode:
                try:
                    selected_modes[m.lower()] = mode_dict[m.lower()]
                except KeyError:
                    raise ValueError(f"Mode '{m}' not found in Bessel dictionary.")

        # --------------------------------------------------
        # Frequency computation
        # --------------------------------------------------
        f_dict = {}

        for radius in r:
            f_dict[f'{radius}'] = {}

            for length in l:

                # ------------------------
                # Waveguide case
                # ------------------------
                if length == 0:

                    for mode_name, j in selected_modes.items():
                        f = (
                                c0 /
                                (2 * np.pi * np.sqrt(epsr * mu0r)) *
                                (j / (radius * 1e-3))
                        )

                        f_dict[f'{radius}'][mode_name] = f

                # ------------------------
                # Cavity case
                # ------------------------
                else:

                    for mode_name, j in selected_modes.items():

                        pol = mode_name[:2].lower()
                        m = int(mode_name[2])
                        n = int(mode_name[3])

                        for p in range(0, p_max + 1):

                            # Enforce physical condition:
                            # TE modes cannot have p=0 in PEC cavity
                            if pol == "te" and p == 0:
                                continue

                            kr = j / (radius * 1e-3)
                            kz = p * np.pi / (length * 1e-3)

                            f = (
                                    c0 /
                                    (2 * np.pi * np.sqrt(epsr * mu0r)) *
                                    np.sqrt(kr ** 2 + kz ** 2)
                            )

                            full_mode = f"{pol}{m}{n}{p}"
                            f_dict[f'{radius}'][full_mode] = f

        return f_dict

    @staticmethod
    def rwg_cutoff(a, b, mn=None, l=0, p=None, epsr=1, mu0r=1):

        if isinstance(a, (float, int)):
            a = [a]
        if isinstance(b, (float, int)):
            b = [b]
        if isinstance(l, (float, int)):
            l = [l]

        if mn is None:
            mn = [[1, 0]]

        if len(np.array(mn).shape) == 1:
            mn = [mn]

        if p is None:
            p = [0]
        elif isinstance(p, int):
            p = [p]

        f_dict = {}

        for a_ in a:
            f_dict[f'a: {a_} mm'] = {}

            for b_ in b:
                f_dict[f'a: {a_} mm'][f'b: {b_} mm'] = {}

                for l_ in l:

                    # ---------------------------
                    # Waveguide case
                    # ---------------------------
                    if l_ == 0:

                        for m, n in mn:

                            kc = np.sqrt(
                                (m * np.pi / (a_ * 1e-3)) ** 2 +
                                (n * np.pi / (b_ * 1e-3)) ** 2
                            )

                            f = (c0 / (2 * np.pi * np.sqrt(epsr * mu0r))) * kc

                            if m == 0 and n == 0:
                                continue

                            mode_name = f"TE/TM({m},{n})"
                            f_dict[f'a: {a_} mm'][f'b: {b_} mm'][mode_name] = f

                    # ---------------------------
                    # Cavity case
                    # ---------------------------
                    else:

                        f_dict[f'a: {a_} mm'][f'b: {b_} mm'][f'l: {l_} mm'] = {}

                        for m, n in mn:
                            for p_ in p:

                                k = np.sqrt(
                                    (m * np.pi / (a_ * 1e-3)) ** 2 +
                                    (n * np.pi / (b_ * 1e-3)) ** 2 +
                                    (p_ * np.pi / (l_ * 1e-3)) ** 2
                                )

                                f = (c0 / (2 * np.pi * np.sqrt(epsr * mu0r))) * k

                                if m == 0 and n == 0 and p_ == 0:
                                    continue

                                mode_name = f"TE/TM({m},{n},{p_})"
                                f_dict[f'a: {a_} mm'][f'b: {b_} mm'][f'l: {l_} mm'][mode_name] = f

        return f_dict

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
        C = eps0*epsr * l * b / d
        return {"C' [pF]": C*1e12}

    @staticmethod
    def parallel_disc_capacitor(D, s, epsr=1):
        D, s = D * 1e-3, s * 1e-3
        C = eps0*epsr * np.pi*D**2/(4*s)*(1 + s/(np.pi*D/2))
        return {"C' [pF]": C*1e12}

    @staticmethod
    def coaxial_capacitor(D, d, h, epsr=1):
        D, d, h = D * 1e-3, d * 1e-3, h*1e-3
        C = 2*np.pi*eps0*epsr*h/ (np.log(D/d))
        return {"C' [pF]": C*1e12}

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

            if k == 0:
                beta = 0
            else:
                # Use complex sqrt to handle k < kc cases
                beta = np.lib.scimath.sqrt(k ** 2 - kc ** 2)

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
            if k == 0:
                beta = 0
            else:
                # Use complex sqrt to handle k < kc cases
                beta = np.lib.scimath.sqrt(k**2 - kc**2)
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
        if k == 0:
            beta = 0
        else:
            # Use complex sqrt to handle k < kc cases
            beta = np.lib.scimath.sqrt(k**2 - kc**2)
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

    def pretty_print_modes(self, data, unit="MHz"):
        for param, modes in data.items():
            print(f"\nParameter: {param}")
            print("-" * 50)
            print(f"{'Mode':<8} {'Frequency (' + unit + ')':>20}")
            print("-" * 50)

            # sort by frequency
            for mode, freq in sorted(modes.items(), key=lambda x: float(x[1])):
                if unit == "MHz":
                    value = float(freq) / 1e6
                elif unit == "GHz":
                    value = float(freq) / 1e9
                else:
                    value = float(freq)

                print(f"{mode:<8} {value:>20.6f}")

            print("-" * 50)
