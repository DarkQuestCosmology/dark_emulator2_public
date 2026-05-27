import sys
import copy
import pprint
import random
import numpy as np

from .class_lin_power import get_omega_nu
from . import utils as ut


class Parameters:
    def __init__(self):
        self.verbose = False
        self.nu_massless = False

        self.fid_param = {"Omega_m": 0.3156, "omega_b": 0.02225, "omega_cdm": 0.1198,
                          "Mnu": 0.06, "Omega_k": 0.0, "Omega_de": 0.68434,
                          "sigma8": 0.831, "ns": 0.9645, "w0": -1, "wa": 0, "h0": 0.6723473803969963,
                          "ln(10^10As)": 3.094, "As": 2.206516233894705e-09}

        self.param_range = {"Omega_m": [0.05, 0.62], "omega_b": [0.015, 0.03], "omega_cdm": [0.01, 0.3],
                            "Mnu": [0.0, 0.5], "Omega_k": [-0.1, 0.1], "Omega_de": [0.4125, 0.938542],
                            "sigma8": [0.42, 1.3], "ns": [0.916275, 1.012725],
                            "w": [-1.5, -0.5], "w0": [-1.5, -0.5], "wa": [-0.5, 0.5], "h0": [0.5, 0.9],
                            "ln(10^10As)": [1.0412, 5.7548], "As": [2.832614112620091e-10, 3.1570240072959045e-08],
                            "S8": [0.6, 0.95], "dist": [0.0, 5.0]}

    @staticmethod
    def show_template():
        """
        Print the required parameter combinations for a valid DQ2 parameter set.
        """
        print("\n")
        print("The 9 - dimensional parameter requires this combination.\n"
              "'ns', 'w0', 'wa', 'Mnu' and \n"
              "one of the following ['sigma8', 'As', 'ln(10^10As)']\n"
              "three of the following ['Omega_m', 'h0', 'omega_b', 'omega_cdm']\n"
              "one of the following ['Omega_k', 'Omega_de']\n")

        print("Example : ")
        print("param={'Omega_m': 0.315, 'omega_b': 0.02225,"
              "'omega_cdm': 0.1198, 'Mnu': 0.06, 'Omega_k': 0.0,"
              "'sigma8': 0.831, 'ns': 0.9645, 'w0': -1, 'wa': 0}")
        print(
            "consistency_param = obj.set_param(param) # <--- Check parameters and return consistency parameter.")

    def get_fid_param(self):
        """
        Returns a deepcopy of the fiducial cosmological parameters.
        """
        return copy.deepcopy(self.fid_param)

    def get_param_range(self):
        """
        Returns a deepcopy of the allowed parameter ranges.
        """
        return copy.deepcopy(self.param_range)

    def check_parameter_range(self, param, strict=True):
        """
        Check if the given parameter dictionary values are within allowed ranges.

        Parameters
        ----------
        param : dict
            Dictionary containing cosmological parameters to check.
        strict : bool, optional
            If True, raises a ValueError when a parameter is out of range.
            If False, prints a warning to standard error and continues.

        Raises
        ------
        ValueError
            If strict is True and a parameter is out of range.
        """

        for idict in self.param_range.keys():
            if idict in param.keys():
                if param[idict] == None:
                    continue
                lo, hi = self.param_range[idict]
                if not (lo <= param[idict] <= hi):
                    msg = f"[Warning] {idict}={param[idict]} is out of range [{lo}, {hi}]"
                    if strict:
                        raise ValueError(msg)
                    else:
                        print(msg, file=sys.stderr)

    @staticmethod
    def def_key_val(p, key):
        if key not in p.keys():
            return False
        if p[key] == None:
            return False
        return True

    def check_key_val_num(self, p, keys):
        check = sum(self.def_key_val(p, ikey) for ikey in keys)
        return check

    def set_param(self, _param, strict=True):
        """
        Check consistency of the parameter dictionary and fill missing values
        based on internal cosmology relations.

        Parameters
        ----------
        _param : dict
            User provided parameter dictionary.
        strict : bool, optional
            If True, raises a ValueError when a parameter is out of the allowed range.
            If False, only prints warnings and continues.

        Returns
        -------
        dict
            Completed and consistent parameter dictionary.


        Notes
        -----
        The 9-dimensional emulator parameter requires the following combination:

        - 'ns', 'w0', 'wa', 'Mnu'
        - one of ['sigma8', 'As', 'ln(10^10As)']
        - three of ['h0', 'omega_b', 'omega_cdm', 'Omega_m']
        - one of ['Omega_k', 'Omega_de']
        """

        param = copy.deepcopy(_param)

        check_param = ['sigma8', 'As', 'ln(10^10As)']
        check = self.check_key_val_num(param, check_param)
        assert check == 1, "Must specify exactly one of " + str(check_param)

        check_param = ['h0', 'omega_b', 'omega_cdm', 'Omega_m']
        check = self.check_key_val_num(param, check_param)
        assert check == 3, "Must specify exactly three of " + str(check_param)

        check_param = ['Omega_k', 'Omega_de']
        check = self.check_key_val_num(param, check_param)
        assert check == 1, "Must specify exactly one of " + str(check_param)

        m_ncdm = param['Mnu'] / 3
        omega_nu = get_omega_nu(m_ncdm)
        oMnu_tot = 3 * omega_nu

        if not self.def_key_val(param, 'h0'):
            param['h0'] = np.sqrt(
                (param['omega_b'] + param['omega_cdm'] + oMnu_tot) / param['Omega_m'])
        else:
            h0 = param['h0']
            if not self.def_key_val(param, 'Omega_m'):
                param['Omega_m'] = (param['omega_b'] +
                                    param['omega_cdm'] + oMnu_tot) / h0**2
            elif not self.def_key_val(param, 'omega_b'):
                param['omega_b'] = param['Omega_m'] * \
                    h0**2 - (param['omega_cdm'] + oMnu_tot)
            elif not self.def_key_val(param, 'omega_cdm'):
                param['omega_cdm'] = param['Omega_m'] * \
                    h0**2 - (param['omega_b'] + oMnu_tot)
            elif not self.def_key_val(param, 'Mnu'):
                raise ValueError("Mnu must be specified or inferrable.")
        assert self.def_key_val(param, 'Omega_m')

        # Omega_m + Omega_k + Omega_de = 1
        check_param = ['Omega_k', 'Omega_de']
        check = self.check_key_val_num(param, check_param)
        assert check >= 1, "Specify one parameters of the following " + check_param

        if not self.def_key_val(param, 'Omega_k'):
            param['Omega_k'] = 1.0 - param['Omega_m'] - param['Omega_de']
        elif not self.def_key_val(param, 'Omega_de'):
            # Not use CLASS input
            param['Omega_de'] = 1.0 - param['Omega_m'] - param['Omega_k']

        self.check_parameter_range(param, strict)

        if self.verbose:
            print("[Parameters] Setting parameters:")
            pprint.pprint(param, sort_dicts=False)

        return param

    @staticmethod
    def __inner_epi(x, y, x0, xr, y0, yr):
        f = (x - x0)**2 / xr**2 + (y - y0)**2 / yr**2 - 1
        a = True if f < 0 else False
        return a

    def __rand_param_basic(self):
        Om_range = self.param_range["Omega_m"]
        S8_range = self.param_range["S8"]

        Om_cent = self.fid_param["Omega_m"]
        sig8_cent = self.fid_param["sigma8"]
        S8_cent = sig8_cent * np.sqrt(Om_cent / 0.3)

        # This can be matched to the input freehand range of DQ2.
        S8_cent *= 0.91

        Om_r = np.fmax(Om_range[1] - Om_cent, Om_cent - Om_range[0])
        S8_r = np.fmax(S8_range[1] - S8_cent, S8_cent - S8_range[0])

        while 1:  # for collect omaga_b
            while 1:  # for collect S8
                Om = random.uniform(*Om_range)
                S8 = random.uniform(*S8_range)

                if self.__inner_epi(Om, S8, Om_cent, Om_r, S8_cent, S8_r):
                    break

            param = {}
            keys = ["omega_cdm", "h0", "Mnu", "ns", "Omega_k", "w0", "wa"]
            for ikey in keys:
                param[ikey] = random.uniform(*self.param_range[ikey])

            if self.nu_massless:
                param['Mnu'] = 0.0
                m_ncdm = param['Mnu'] / 3
                omega_nu = 0.0
            else:
                m_ncdm = param['Mnu'] / 3
                omega_nu = get_omega_nu(m_ncdm)

            oMnu_tot = 3 * omega_nu
            param["omega_b"] = Om * param["h0"]**2 - param["omega_cdm"] - oMnu_tot

            if ((param["omega_b"] >= self.param_range["omega_b"][0]) and
                    (param["omega_b"] <= self.param_range["omega_b"][1])):
                break

        # S8 = sigma8 * (Om/0.3)**0.5
        sig8 = S8 / (Om / 0.3)**0.5
        param["sigma8"] = sig8
        param["Omega_m"] = Om
        param["S8"] = S8

        self.clear_As(param)
        self.clear_omega_cdm(param)
        param = self.set_param(param)
        return param

    def __rand_param_with_range(self, range_dict):

        Om_range = self.param_range["Omega_m"]
        S8_range = self.param_range["S8"]

        Om_cent = self.fid_param["Omega_m"]
        sig8_cent = self.fid_param["sigma8"]
        S8_cent = sig8_cent * np.sqrt(Om_cent / 0.3)

        # This can be matched to the input freehand range of DQ2.
        S8_cent *= 0.91

        Om_r = np.fmax(Om_range[1] - Om_cent, Om_cent - Om_range[0])
        S8_r = np.fmax(S8_range[1] - S8_cent, S8_cent - S8_range[0])

        while 1:  # for collect sigma8
            while 1:  # for collect omaga_b
                while 1:  # for collect S8
                    if "Omega_m" in range_dict:
                        Om = random.uniform(*range_dict["Omega_m"])
                    else:
                        Om = random.uniform(*Om_range)

                    if "S8" in range_dict:
                        S8 = random.uniform(*range_dict["S8"])
                    else:
                        S8 = random.uniform(*S8_range)

                    if self.__inner_epi(Om, S8, Om_cent, Om_r, S8_cent, S8_r):
                        break

                param = {}
                keys = ["omega_cdm", "h0", "Mnu", "ns", "Omega_k", "w0", "wa"]
                for ikey in keys:
                    if ikey in range_dict:
                        param[ikey] = random.uniform(*range_dict[ikey])
                    else:
                        param[ikey] = random.uniform(*self.param_range[ikey])

                if self.nu_massless:
                    param['Mnu'] = 0.0
                    m_ncdm = 0.0
                    omega_nu = 0.0
                else:
                    m_ncdm = param['Mnu'] / 3
                    omega_nu = get_omega_nu(m_ncdm)

                m_ncdm = param['Mnu'] / 3
                omega_nu = get_omega_nu(m_ncdm)
                oMnu_tot = 3 * omega_nu
                param["omega_b"] = Om * param["h0"]**2 - param["omega_cdm"] - oMnu_tot

                if "omega_b" in range_dict:
                    if ((param["omega_b"] >= range_dict["omega_b"][0]) and
                            (param["omega_b"] <= range_dict["omega_b"][1])):
                        break
                else:
                    if ((param["omega_b"] >= self.param_range["omega_b"][0]) and
                            (param["omega_b"] <= self.param_range["omega_b"][1])):
                        break

            # S8 = sigma8 * (Om/0.3)**0.5
            sig8 = S8 / (Om / 0.3)**0.5
            param["sigma8"] = sig8
            param["Omega_m"] = Om
            param["S8"] = S8

            if "sigma8" in range_dict:
                if ((param["sigma8"] >= range_dict["sigma8"][0]) and
                        (param["sigma8"] <= range_dict["sigma8"][1])):
                    break
            else:
                if ((param["sigma8"] >= self.param_range["sigma8"][0]) and
                        (param["sigma8"] <= self.param_range["sigma8"][1])):
                    break

        self.clear_As(param)
        self.clear_omega_cdm(param)
        param = self.set_param(param)
        return param

    def rand_seed(self, seed=None):
        """
        Set random seed for reproducibility.

        Parameters
        ----------
        seed : int, optional
            Seed value.
        """

        random.seed(seed)

    def rand_param(self, range_dict=None):
        """
        Generate a random parameter dictionary consistent with constraints.

        Parameters
        ----------
        range_dict : dict, optional
            User-specified subranges for parameters.

        Returns
        -------
        dict
            Random cosmological parameter dictionary.
        """

        if range_dict == None:
            try:
                p = self.__rand_param_basic()
            except (AssertionError, ValueError):
                p = self.rand_param()
        else:
            try:
                p = self.__rand_param_with_range(range_dict)
            except (AssertionError, ValueError):
                p = self.rand_param(range_dict)
        return p

    def rand_param_LCDM(self, range_dict=None):
        """
        Generate a random parameter set for standard LCDM (massless neutrino, w0=-1, wa=0).

        Parameters
        ----------
        range_dict : dict, optional
            Subranges to restrict parameter generation.

        Returns
        -------
        dict
            Random LCDM parameter set.
        """

        self.nu_massless = True
        p = self.rand_param(range_dict)
        self.nu_massless = False  # reset param

        p["w0"] = -1.0
        p["wa"] = 0.0

        # p["Omega_m"] + p["Omega_de"] + p["Omega_k"] = 1.0
        p["Omega_de"] += p["Omega_k"]
        p["Omega_k"] = 0.0
        return p

    def rand_param_nuLCDM(self, range_dict=None):
        """
        Generate a random parameter set for nuLCDM (massive neutrinos, w0=-1, wa=0).
        """

        p = self.rand_param(range_dict)
        p["w0"] = -1.0
        p["wa"] = 0.0
        # p["Omega_m"] + p["Omega_de"] + p["Omega_k"] = 1.0
        p["Omega_de"] += p["Omega_k"]
        p["Omega_k"] = 0.0
        return p

    def rand_param_wCDM(self, range_dict=None):
        """
        Generate a random parameter set for wCDM (massless neutrinos, free w0, wa=0).
        """

        self.nu_massless = True
        p = self.rand_param(range_dict)
        self.nu_massless = False  # reset param

        p["wa"] = 0.0

        # p["Omega_m"] + p["Omega_de"] + p["Omega_k"] = 1.0
        p["Omega_de"] += p["Omega_k"]
        p["Omega_k"] = 0.0
        return p

    def rand_param_w0waCDM(self, range_dict=None):
        """
        Generate a random parameter set for w0waCDM (massless neutrinos, free w0 and wa).
        """

        self.nu_massless = True
        p = self.rand_param(range_dict)
        self.nu_massless = False  # reset param

        # p["Omega_m"] + p["Omega_de"] + p["Omega_k"] = 1.0
        p["Omega_de"] += p["Omega_k"]
        p["Omega_k"] = 0.0
        return p

    def rand_param_nuwCDM(self, range_dict=None):
        """
        Generate a random parameter set for nuwCDM (massive neutrinos, free w0, wa=0).
        """

        p = self.rand_param(range_dict)
        p["wa"] = 0.0
        # p["Omega_m"] + p["Omega_de"] + p["Omega_k"] = 1.0
        p["Omega_de"] += p["Omega_k"]
        p["Omega_k"] = 0.0
        return p

    def rand_param_nuw0waCDM(self, range_dict=None):
        """
        Generate a random parameter set for nuw0waCDM (massive neutrinos, free w0 and wa).
        """

        p = self.rand_param(range_dict)
        # p["Omega_m"] + p["Omega_de"] + p["Omega_k"] = 1.0
        p["Omega_de"] += p["Omega_k"]
        p["Omega_k"] = 0.0
        return p

    def rand_param_w0waoCDM(self, range_dict=None):
        """
        Generate a random parameter set for w0waoCDM
        (massless neutrinos, free w0, wa and curvature).
        """

        self.nu_massless = True
        p = self.rand_param(range_dict)
        self.nu_massless = False  # reset param

        return p

    def rand_param_nuw0waoCDM(self, range_dict=None):
        """
        Generate a random parameter set for the most general case
        (massive neutrinos, free w0, wa and curvature).
        """

        p = self.rand_param(range_dict)
        return p

    def rand_param_cosmology(self, nu=True, w0=True, wa=True, K=True, range_dict=None):
        """
        Dispatch function to generate random parameters for a cosmology model.

        Parameters
        ----------
        nu : bool
            Include massive neutrinos if True.
        w0 : bool
            Include w0 parameter if True.
        wa : bool
            Include wa parameter if True.
        K : bool
            Include curvature if True.
        range_dict : dict, optional
            Parameter subranges.

        Returns
        -------
        dict
            Random parameter set matching the specified cosmology.
        """

        key = (nu, w0, wa, K)
        dispatch = {
            (True, True, True, True): self.rand_param_nuw0waoCDM,
            (True, True, True, False): self.rand_param_nuw0waCDM,
            (True, True, False, False): self.rand_param_nuwCDM,
            (True, False, False, False): self.rand_param_nuLCDM,
            (False, True, True, True): self.rand_param_w0waoCDM,
            (False, True, True, False): self.rand_param_w0waCDM,
            (False, True, False, False): self.rand_param_wCDM,
            (False, False, False, False): self.rand_param_LCDM,
        }

        if self.verbose:
            print(f"[Parameters] Selected cosmology type for nu={nu}, w0={w0}, wa={wa}, K={K}")

        if key not in dispatch:
            raise ValueError(f"Unsupported cosmology combination: nu={nu}, w0={w0}, wa={wa}, K={K}")
        return dispatch[key](range_dict)

    def set_params_from_file_to_dict(self, param_file):
        """
        Load parameters from a text file and convert to dictionary.

        Parameters
        ----------
        param_file : str
            Path to parameter file.

        Returns
        -------
        dict
            Dictionary of parameters.

        Notes
        -----
        The parameter file must be a whitespace-delimited text file with a header line
        specifying parameter names. For example:

            # Omega_m omega_b omega_cdm Mnu Omega_k sigma8 ns w0 wa h0
            0.3156  0.02225 0.1198    0.06 0.0    0.831 0.9645 -1 0 0.6723
            ...
            ...
            ...

        The first line is interpreted as column names (header).
        Subsequent lines are treated as data (only first row is loaded here).
        """

        header = ut.read_header_label(param_file)
        param_dtype = np.dtype({'names': header, 'formats': (np.float64,) * len(header)})
        p = np.loadtxt(param_file, dtype=param_dtype)
        p = ut.sarray_to_dict(p)
        return p

    def __de2_om_sig8_support_params(self):
        Om_range = [0.1, 0.6]
        sig8_max = 1.25
        S8_range = [0.6, 0.95]
        Om_cent = 0.3156
        sig8_cent = 0.831
        S8_cent = sig8_cent * np.sqrt(Om_cent / 0.3)
        S8_cent *= 0.91
        Om_r = np.fmax(Om_range[1] - Om_cent, Om_cent - Om_range[0])
        S8_r = np.fmax(S8_range[1] - S8_cent, S8_cent - S8_range[0])
        return Om_cent, Om_r, S8_cent, S8_r, sig8_max

    def get_de2_om_sig8_plane(self):
        Om_cent, Om_r, S8_cent, S8_r, sig8_max = self.__de2_om_sig8_support_params()
        theta = np.linspace(0, 2 * np.pi, 256)
        x = Om_r * np.cos(theta) + Om_cent
        y = S8_r * np.sin(theta) + S8_cent
        y2 = y / (x / 0.3)**0.5
        de2_range_x = x[y2 < sig8_max * 1.01]
        de2_range_y = y2[y2 < sig8_max * 1.01]
        return de2_range_x, de2_range_y

    def _is_in_Om_sig8_support(self, param):
        """
        Check the approximate DE2 support in the Omega_m-sigma8 plane.

        This uses the same projected support definition as
        ``get_de2_om_sig8_plane()``. It only checks the projected banana
        with the high-sigma8 clip, not the full 9D parameter consistency.
        """
        Om = param["Omega_m"]

        if self.def_key_val(param, "sigma8"):
            sig8 = param["sigma8"]
            S8 = sig8 * np.sqrt(Om / 0.3)
        elif self.def_key_val(param, "S8"):
            S8 = param["S8"]
            sig8 = S8 / np.sqrt(Om / 0.3)
        else:
            raise KeyError("param must contain 'Omega_m' and either 'sigma8' or 'S8'")

        Om_cent, Om_r, S8_cent, S8_r, sig8_max = self.__de2_om_sig8_support_params()
        ellipse = (Om - Om_cent)**2 / Om_r**2 + (S8 - S8_cent)**2 / S8_r**2
        return bool((ellipse <= 1.0) and (sig8 <= sig8_max * 1.01))

    def clear_sigma8(self, param):
        """
        Remove ``sigma8`` from a parameter dictionary if present.
        """
        param.pop("sigma8", None)

    def clear_As(self, param):
        """
        Remove ``As`` from a parameter dictionary if present.
        """
        param.pop("As", None)

    def clear_lnAs(self, param):
        """
        Remove ``ln(10^10As)`` from a parameter dictionary if present.
        """
        param.pop("ln(10^10As)", None)

    def clear_Omega_m(self, param):
        """
        Remove ``Omega_m`` from a parameter dictionary if present.
        """
        param.pop("Omega_m", None)

    def clear_omega_cdm(self, param):
        """
        Remove ``omega_cdm`` from a parameter dictionary if present.
        """
        param.pop("omega_cdm", None)

    def clear_omega_b(self, param):
        """
        Remove ``omega_b`` from a parameter dictionary if present.
        """
        param.pop("omega_b", None)

    def clear_Omega_de(self, param):
        """
        Remove ``Omega_de`` from a parameter dictionary if present.
        """
        param.pop("Omega_de", None)

    def clear_Omega_k(self, param):
        """
        Remove ``Omega_k`` from a parameter dictionary if present.
        """
        param.pop("Omega_k", None)

    def clear_h0(self, param):
        """
        Remove ``h0`` from a parameter dictionary if present.
        """
        param.pop("h0", None)

    ############################################################################
    # preset public emulator parameter ranges
    ############################################################################
    def _clip_to_dq2_range(self, vmin, vmax, key):
        if key not in self.param_range:
            # No global prior for this variable; keep original.
            return vmin, vmax

        lo, hi = self.param_range[key]
        vmin_clipped = max(vmin, lo)
        vmax_clipped = min(vmax, hi)

        if vmin_clipped > vmax_clipped:
            raise ValueError(
                f"No overlap between emulator prior [{vmin}, {vmax}] and "
                f"DQ2 prior [{lo}, {hi}] for '{key}'."
            )
        return vmin_clipped, vmax_clipped

    def get_preset_range_dict(self, name=None):
        """
        Map external emulator parameter ranges to a rectangular prior
        in the DQ2 native parameter space.

        Parameters
        ----------
        name : str
            Preset name, case-insensitive. The ``preset_`` prefix is optional.

        Notes
        -----
        Accepted preset names are:

        - ``preset_de2``: DQ2 native emulator range
        - ``preset_ee2``: EuclidEmulator2 <https://arxiv.org/abs/2010.11288>
        - ``preset_bacco``: BACCOemu <https://arxiv.org/abs/2011.15018>
        - ``preset_aemulus``: Aemulus nonlinear emulator <https://arxiv.org/abs/2303.09762>
        - ``preset_mtu``: MTU emulator <https://arxiv.org/abs/2207.12345>
        - ``preset_goku``: GOKU-W <https://arxiv.org/abs/2501.06296>
        - ``preset_csstemu``: CSSTemu <https://github.com/czymh/csstemu>
        - ``preset_aletheia``: <https://arxiv.org/abs/2511.13826>
        - ``preset_franken``: FrankenEmu <https://arxiv.org/abs/0912.4490>
        - ``preset_pkann``: PKANN <https://arxiv.org/abs/1312.2101>
        - ``preset_hmcode``: HMcode (uses MTU-like box)

        Returns
        -------
        dict
            Rectangular range_dict in terms of DQ2 parameter names.
        """

        if name is None:
            print(f"No preset name provided {name}.")
            print("Available presets: 'preset_de2', 'preset_ee2', 'preset_bacco', 'preset_aemulus', 'preset_mtu', 'preset_goku', 'preset_csstemu', 'preset_aletheia', 'preset_franken', 'preset_pkann', 'preset_hmcode'.")
            return {}

        key = name.lower()
        if not key.startswith("preset_"):
            key = f"preset_{key}"

        if key == "preset_de2":
            return copy.deepcopy(self.param_range)

        # --------------------------------------------------------------
        # EuclidEmulator2 (EE2):
        #   Omega_b  in [0.04, 0.06]
        #   Omega_m  in [0.24, 0.40]
        #   m_ncdm   in [0.0, 0.15]   (sum m_nu)
        #   n_s      in [0.92, 1.0]
        #   h        in [0.61, 0.73]
        #   w0_fld   in [-1.3, -0.7]
        #   wa_fld   in [-0.7, 0.5]
        #   A_s      in [1.7e-9, 2.5e-9]
        # Here we convert:
        #   Omega_b -> omega_b (physical) using emulator h-range,
        #   m_ncdm  -> Mnu,
        #   and clip all to the DQ2 global box.
        # --------------------------------------------------------------
        if key == "preset_ee2":
            # Emulator ranges
            Omb_min, Omb_max = 0.04, 0.06
            Om_min, Om_max = 0.24, 0.40
            As_min, As_max = 1.7e-9, 2.5e-9
            ns_min, ns_max = 0.92, 1.0
            h_min, h_max = 0.61, 0.73
            w0_min, w0_max = -1.3, -0.7
            wa_min, wa_max = -0.7, 0.5
            Mnu_min, Mnu_max = 0.0, 0.15

            # Clip to DQ2 global box
            Om_min, Om_max = self._clip_to_dq2_range(Om_min, Om_max, "Omega_m")
            h_min, h_max = self._clip_to_dq2_range(h_min, h_max, "h0")
            As_min, As_max = self._clip_to_dq2_range(As_min, As_max, "As")
            ns_min, ns_max = self._clip_to_dq2_range(ns_min, ns_max, "ns")
            w0_min, w0_max = self._clip_to_dq2_range(w0_min, w0_max, "w0")
            wa_min, wa_max = self._clip_to_dq2_range(wa_min, wa_max, "wa")

            omega_b_min = Omb_min * (h_min ** 2)
            omega_b_max = Omb_max * (h_max ** 2)
            omega_b_min, omega_b_max = self._clip_to_dq2_range(omega_b_min, omega_b_max, "omega_b")
            Mnu_min, Mnu_max = self._clip_to_dq2_range(Mnu_min, Mnu_max, "Mnu")

            return {
                "Omega_m": [Om_min, Om_max],
                "omega_b": [omega_b_min, omega_b_max],
                "As": [As_min, As_max],
                "ns": [ns_min, ns_max],
                "h0": [h_min, h_max],
                "Mnu": [Mnu_min, Mnu_max],
                "w0": [w0_min, w0_max],
                "wa": [wa_min, wa_max],
            }

        # --------------------------------------------------------------
        # BACCOemu (baccoemu.Matter_powerspectrum):
        #   omega_cold   ~ Omega_cb in [0.23, 0.40]
        #   omega_baryon ~ Omega_b  in [0.04, 0.06]
        #   ns           in [0.92, 1.01]
        #   hubble       in [0.6, 0.8]
        #   neutrino_mass (sum m_nu) in [0.0, 0.4]
        #   w0           in [-1.15, -0.85]
        #   wa           in [-0.3, 0.3]
        #   sigma8_cold  in [0.73, 0.9]
        #
        # We only map:
        #   Omega_b  -> omega_b   (via h-range),
        #   neutrino_mass -> Mnu,
        #   hubble  -> h0,
        #   and (ns, w0, wa).
        # We do not try to represent the combined constraint on Omega_cb
        # or sigma8_cold here.
        # --------------------------------------------------------------
        if key == "preset_bacco":
            Ob_min, Ob_max = 0.04, 0.06
            h_min, h_max = 0.6, 0.8
            ns_min, ns_max = 0.92, 1.01
            Mnu_min, Mnu_max = 0.0, 0.4
            w0_min, w0_max = -1.15, -0.85
            wa_min, wa_max = -0.3, 0.3

            h_min, h_max = self._clip_to_dq2_range(h_min, h_max, "h0")
            ns_min, ns_max = self._clip_to_dq2_range(ns_min, ns_max, "ns")
            Mnu_min, Mnu_max = self._clip_to_dq2_range(Mnu_min, Mnu_max, "Mnu")
            w0_min, w0_max = self._clip_to_dq2_range(w0_min, w0_max, "w0")
            wa_min, wa_max = self._clip_to_dq2_range(wa_min, wa_max, "wa")

            omega_b_min = Ob_min * (h_min ** 2)
            omega_b_max = Ob_max * (h_max ** 2)
            omega_b_min, omega_b_max = self._clip_to_dq2_range(omega_b_min, omega_b_max, "omega_b")

            return {
                "omega_b": [omega_b_min, omega_b_max],
                "ns": [ns_min, ns_max],
                "h0": [h_min, h_max],
                "Mnu": [Mnu_min, Mnu_max],
                "w0": [w0_min, w0_max],
                "wa": [wa_min, wa_max],
            }

        # --------------------------------------------------------------
        # Aemulus nonlinear power spectrum emulator:
        #   omega_b  in [0.0173, 0.0272]
        #   omega_cdm in [0.08, 0.16]
        #   Mnu      in [0.01, 0.5]
        #   ns       in [0.93, 1.01]
        #   h0       in [0.52, 0.82]
        #   w0       in [-1.56, -0.44]
        #   As       in [1.1e-9, 3.1e-9]
        # --------------------------------------------------------------
        if key == "preset_aemulus":
            omega_b_min, omega_b_max = 0.0173, 0.0272
            omega_cdm_min, omega_cdm_max = 0.08, 0.16
            Mnu_min, Mnu_max = 0.01, 0.5
            ns_min, ns_max = 0.93, 1.01
            h_min, h_max = 0.52, 0.82
            w0_min, w0_max = -1.56, -0.44
            As_min, As_max = 1.1e-9, 3.1e-9

            omega_b_min, omega_b_max = self._clip_to_dq2_range(omega_b_min, omega_b_max, "omega_b")
            omega_cdm_min, omega_cdm_max = self._clip_to_dq2_range(omega_cdm_min, omega_cdm_max, "omega_cdm")
            Mnu_min, Mnu_max = self._clip_to_dq2_range(Mnu_min, Mnu_max, "Mnu")
            ns_min, ns_max = self._clip_to_dq2_range(ns_min, ns_max, "ns")
            h_min, h_max = self._clip_to_dq2_range(h_min, h_max, "h0")
            w0_min, w0_max = self._clip_to_dq2_range(w0_min, w0_max, "w0")
            As_min, As_max = self._clip_to_dq2_range(As_min, As_max, "As")

            return {
                "omega_b": [omega_b_min, omega_b_max],
                "omega_cdm": [omega_cdm_min, omega_cdm_max],
                "Mnu": [Mnu_min, Mnu_max],
                "ns": [ns_min, ns_max],
                "h0": [h_min, h_max],
                "w0": [w0_min, w0_max],
                "As": [As_min, As_max],
            }

        # --------------------------------------------------------------
        # MTU emulator:
        #   omega_m  in [0.120, 0.155]   (omega_m = Omega_m * h^2)
        #   sigma8   in [0.7, 0.9]
        #   omega_b  in [0.0215, 0.0235]
        #   ns       in [0.85, 1.05]
        #   h        in [0.55, 0.85]
        #   omega_nu in [0.0, 0.01]     (physical neutrino density)
        #   w0       in [-1.3, -0.7]
        #   wa       in [-1.5, 1.15]
        #   -(w0+wa)^(1/4) in [0.3, 1.29]  (not used here)
        #
        # We map:
        #   omega_b -> omega_b,
        #   sigma8  -> sigma8,
        #   ns, h, w0, wa,
        #   omega_nu -> Mnu via Mnu ≈ 93.14 * omega_nu.
        # For omega_m we derive a projected Omega_m range:
        #   Omega_m_min = omega_m_min / h_max^2
        #   Omega_m_max = omega_m_max / h_min^2
        # --------------------------------------------------------------
        if key == "preset_mtu":
            omega_m_min, omega_m_max = 0.120, 0.155
            sigma8_min, sigma8_max = 0.7, 0.9
            omega_b_min, omega_b_max = 0.0215, 0.0235
            ns_min, ns_max = 0.85, 1.05
            h_min, h_max = 0.55, 0.85
            omega_nu_min, omega_nu_max = 0.0, 0.01
            w0_min, w0_max = -1.3, -0.7
            wa_min, wa_max = -1.5, 1.15

            h_min, h_max = self._clip_to_dq2_range(h_min, h_max, "h0")

            # Derived Omega_m range from omega_m and h-range.
            Om_min = omega_m_min / (h_max ** 2)
            Om_max = omega_m_max / (h_min ** 2)

            Om_min, Om_max = self._clip_to_dq2_range(Om_min, Om_max, "Omega_m")
            sigma8_min, sigma8_max = self._clip_to_dq2_range(sigma8_min, sigma8_max, "sigma8")
            omega_b_min, omega_b_max = self._clip_to_dq2_range(omega_b_min, omega_b_max, "omega_b")
            ns_min, ns_max = self._clip_to_dq2_range(ns_min, ns_max, "ns")
            w0_min, w0_max = self._clip_to_dq2_range(w0_min, w0_max, "w0")
            wa_min, wa_max = self._clip_to_dq2_range(wa_min, wa_max, "wa")

            # Approximate conversion omega_nu -> Mnu
            Mnu_min = 93.14 * omega_nu_min
            Mnu_max = 93.14 * omega_nu_max
            Mnu_min, Mnu_max = self._clip_to_dq2_range(Mnu_min, Mnu_max, "Mnu")

            return {
                "Omega_m": [Om_min, Om_max],
                "omega_b": [omega_b_min, omega_b_max],
                "sigma8": [sigma8_min, sigma8_max],
                "ns": [ns_min, ns_max],
                "h0": [h_min, h_max],
                "w0": [w0_min, w0_max],
                "wa": [wa_min, wa_max],
                "Mnu": [Mnu_min, Mnu_max],
            }

        # --------------------------------------------------------------
        # GOKU-W:
        #   Omega_m in [0.22, 0.40]
        #   h0      in [0.60, 0.76]
        #   Omega_b in [0.040, 0.055]
        #   ns      in [0.80, 1.10]
        #   As      in [1.0e-9, 3.0e-9]
        #   Mnu     in [0.0, 0.6]
        #   w0      in [-1.30, 0.25]
        #   wa      in [-3.0, 0.5]
        #   Neff    in [2.2, 4.5]      (not used here)
        #   alpha_s in [-0.05, 0.05]   (not used here)
        #
        # We convert:
        #   Omega_b -> omega_b via emulator h-range,
        #   and keep all DQ2-native parameters.
        # --------------------------------------------------------------
        if key == "preset_goku":
            Om_min, Om_max = 0.22, 0.40
            Ob_min, Ob_max = 0.040, 0.055
            h_min, h_max = 0.60, 0.76
            ns_min, ns_max = 0.80, 1.10
            As_min, As_max = 1.0e-9, 3.0e-9
            Mnu_min, Mnu_max = 0.0, 0.6
            w0_min, w0_max = -1.30, 0.25
            wa_min, wa_max = -3.0, 0.5

            h_min, h_max = self._clip_to_dq2_range(h_min, h_max, "h0")

            omega_b_min = Ob_min * (h_min ** 2)
            omega_b_max = Ob_max * (h_max ** 2)

            Om_min, Om_max = self._clip_to_dq2_range(Om_min, Om_max, "Omega_m")
            omega_b_min, omega_b_max = self._clip_to_dq2_range(omega_b_min, omega_b_max, "omega_b")
            ns_min, ns_max = self._clip_to_dq2_range(ns_min, ns_max, "ns")
            As_min, As_max = self._clip_to_dq2_range(As_min, As_max, "As")
            Mnu_min, Mnu_max = self._clip_to_dq2_range(Mnu_min, Mnu_max, "Mnu")
            w0_min, w0_max = self._clip_to_dq2_range(w0_min, w0_max, "w0")
            wa_min, wa_max = self._clip_to_dq2_range(wa_min, wa_max, "wa")

            return {
                "Omega_m": [Om_min, Om_max],
                "omega_b": [omega_b_min, omega_b_max],
                "h0": [h_min, h_max],
                "ns": [ns_min, ns_max],
                "As": [As_min, As_max],
                "Mnu": [Mnu_min, Mnu_max],
                "w0": [w0_min, w0_max],
                "wa": [wa_min, wa_max],
            }

        # --------------------------------------------------------------
        # CSSTemu:
        #   Omega_b  in [0.04, 0.06]
        #   Omega_cb in [0.24, 0.40]
        #   H0       in [60, 80]
        #   ns       in [0.92, 1.00]
        #   As       in [1.7e-9, 2.5e-9]
        #   w0       in [-1.3, -0.7]
        #   wa       in [-0.5, 0.5]
        #   Mnu      in [0.0, 0.3]
        #
        # We convert:
        #   Omega_b -> omega_b via emulator h-range,
        #   Omega_cb -> projected DQ2 total Omega_m range by adding
        #   an approximate maximum Omega_nu. Exact Omega_b and Omega_cb
        #   bounds should still be checked by the caller.
        # --------------------------------------------------------------
        if key == "preset_csstemu":
            Ob_min, Ob_max = 0.04, 0.06
            Ocb_min, Ocb_max = 0.24, 0.40
            h_min, h_max = 0.60, 0.80
            ns_min, ns_max = 0.92, 1.00
            As_min, As_max = 1.7e-9, 2.5e-9
            w0_min, w0_max = -1.3, -0.7
            wa_min, wa_max = -0.5, 0.5
            Mnu_min, Mnu_max = 0.0, 0.3

            h_min, h_max = self._clip_to_dq2_range(h_min, h_max, "h0")

            omega_b_min = Ob_min * (h_min ** 2)
            omega_b_max = Ob_max * (h_max ** 2)
            omega_nu_max = Mnu_max / (93.14 * h_min ** 2)
            Om_min = Ocb_min
            Om_max = Ocb_max + omega_nu_max

            Om_min, Om_max = self._clip_to_dq2_range(Om_min, Om_max, "Omega_m")
            omega_b_min, omega_b_max = self._clip_to_dq2_range(omega_b_min, omega_b_max, "omega_b")
            ns_min, ns_max = self._clip_to_dq2_range(ns_min, ns_max, "ns")
            As_min, As_max = self._clip_to_dq2_range(As_min, As_max, "As")
            Mnu_min, Mnu_max = self._clip_to_dq2_range(Mnu_min, Mnu_max, "Mnu")
            w0_min, w0_max = self._clip_to_dq2_range(w0_min, w0_max, "w0")
            wa_min, wa_max = self._clip_to_dq2_range(wa_min, wa_max, "wa")

            return {
                "Omega_m": [Om_min, Om_max],
                "omega_b": [omega_b_min, omega_b_max],
                "h0": [h_min, h_max],
                "ns": [ns_min, ns_max],
                "As": [As_min, As_max],
                "Mnu": [Mnu_min, Mnu_max],
                "w0": [w0_min, w0_max],
                "wa": [wa_min, wa_max],
            }

        # --------------------------------------------------------------
        # Aletheia nonlinear power spectrum emulator:
        #   ale_emu = AletheiaEmu()
        #   mu = ale_emu.planck_means       # (3,)
        #   V = ale_emu.eigenvecs          # (3,3)
        #   sig = ale_emu.planck_sigmas      # (3,)
        #   L = 5.0 * np.sum(np.abs(V) * sig[None, :], axis=1)
        #   min_vals = mu - L
        #   max_vals = mu + L
        # --------------------------------------------------------------
        if key == "preset_aletheia":
            omega_b_min, omega_b_max = 0.02113895, 0.02358433
            omega_cdm_min, omega_cdm_max = 0.11099593, 0.13042411
            ns_min, ns_max = 0.94192937, 0.98766975
            Mnu_min, Mnu_max = 0.0, 0.0  # current version

            omega_b_min, omega_b_max = self._clip_to_dq2_range(omega_b_min, omega_b_max, "omega_b")
            omega_cdm_min, omega_cdm_max = self._clip_to_dq2_range(omega_cdm_min, omega_cdm_max, "omega_cdm")
            Mnu_min, Mnu_max = self._clip_to_dq2_range(Mnu_min, Mnu_max, "Mnu")
            ns_min, ns_max = self._clip_to_dq2_range(ns_min, ns_max, "ns")

            return {
                "omega_b": [omega_b_min, omega_b_max],
                "omega_cdm": [omega_cdm_min, omega_cdm_max],
                "Mnu": [Mnu_min, Mnu_max],
                "ns": [ns_min, ns_max],
            }

        # --------------------------------------------------------------
        # HMcode:
        # --------------------------------------------------------------
        if key == "preset_hmcode":
            return self.get_preset_range_dict("preset_mtu")

        # --------------------------------------------------------------
        # FrankenEmu:
        #   omega_m in [0.120, 0.155]    (omega_m = Omega_m * h^2)
        #   sigma8  in [0.61, 0.9]
        #   omega_b in [0.0215, 0.0235]
        #   ns      in [0.85, 1.05]
        #   h       in [0.55, 0.85]
        #   w0      in [-1.3, -0.7]
        #
        # We map:
        #   omega_m -> Omega_m via projection,
        #   and (omega_b, sigma8, ns, h, w0).
        # --------------------------------------------------------------
        if key == "preset_franken":
            omega_m_min, omega_m_max = 0.120, 0.155
            sigma8_min, sigma8_max = 0.61, 0.9
            omega_b_min, omega_b_max = 0.0215, 0.0235
            ns_min, ns_max = 0.85, 1.05
            h_min, h_max = 0.55, 0.85
            w0_min, w0_max = -1.3, -0.7

            h_min, h_max = self._clip_to_dq2_range(h_min, h_max, "h0")
            Om_min = omega_m_min / (h_max ** 2)
            Om_max = omega_m_max / (h_min ** 2)

            Om_min, Om_max = self._clip_to_dq2_range(Om_min, Om_max, "Omega_m")
            sigma8_min, sigma8_max = self._clip_to_dq2_range(sigma8_min, sigma8_max, "sigma8")
            omega_b_min, omega_b_max = self._clip_to_dq2_range(omega_b_min, omega_b_max, "omega_b")
            ns_min, ns_max = self._clip_to_dq2_range(ns_min, ns_max, "ns")
            w0_min, w0_max = self._clip_to_dq2_range(w0_min, w0_max, "w0")

            return {
                "Omega_m": [Om_min, Om_max],
                "omega_b": [omega_b_min, omega_b_max],
                "sigma8": [sigma8_min, sigma8_max],
                "ns": [ns_min, ns_max],
                "h0": [h_min, h_max],
                "w0": [w0_min, w0_max],
            }

        # --------------------------------------------------------------
        # PKANN:
        #   omega_m in [0.11, 0.165]   (omega_m = Omega_m * h^2)
        #   sigma8  in [0.6, 0.95]
        #   omega_b in [0.021, 0.024]
        #   ns      in [0.85, 1.05]
        #   Mnu     in [0.0, 1.1]
        #   w0      in [-1.35, -0.65]
        #
        # Again we project omega_m to Omega_m using the PKANN h-range.
        # PKANN uses roughly the same h-range as other emulators;
        # here we adopt [0.55, 0.85] for consistency.
        # --------------------------------------------------------------
        if key == "preset_pkann":
            omega_m_min, omega_m_max = 0.11, 0.165
            sigma8_min, sigma8_max = 0.6, 0.95
            omega_b_min, omega_b_max = 0.021, 0.024
            ns_min, ns_max = 0.85, 1.05
            Mnu_min, Mnu_max = 0.0, 1.1
            w0_min, w0_max = -1.35, -0.65
            h_min, h_max = 0.55, 0.85  # approximate PKANN h box

            h_min, h_max = self._clip_to_dq2_range(h_min, h_max, "h0")
            Om_min = omega_m_min / (h_max ** 2)
            Om_max = omega_m_max / (h_min ** 2)

            Om_min, Om_max = self._clip_to_dq2_range(Om_min, Om_max, "Omega_m")
            sigma8_min, sigma8_max = self._clip_to_dq2_range(sigma8_min, sigma8_max, "sigma8")
            omega_b_min, omega_b_max = self._clip_to_dq2_range(omega_b_min, omega_b_max, "omega_b")
            ns_min, ns_max = self._clip_to_dq2_range(ns_min, ns_max, "ns")
            Mnu_min, Mnu_max = self._clip_to_dq2_range(Mnu_min, Mnu_max, "Mnu")
            w0_min, w0_max = self._clip_to_dq2_range(w0_min, w0_max, "w0")

            return {
                "Omega_m": [Om_min, Om_max],
                "omega_b": [omega_b_min, omega_b_max],
                "sigma8": [sigma8_min, sigma8_max],
                "ns": [ns_min, ns_max],
                "Mnu": [Mnu_min, Mnu_max],
                "w0": [w0_min, w0_max],
                "h0": [h_min, h_max],
            }

        print(f"No preset name provided {name}.")
        print("Available presets: 'preset_de2', 'preset_ee2', 'preset_bacco', 'preset_aemulus', 'preset_mtu', 'preset_goku', 'preset_csstemu', 'preset_aletheia', 'preset_franken', 'preset_pkann', 'preset_hmcode'.")
        return {}

    def get_preset_range_dict_pair(self, name1=None, name2=None):
        """
        Return the rectangular intersection of one or two emulator presets
        in the DQ2 native parameter space.
        """

        if name1 is None:
            print(f"No preset name provided {name1}.")
            print("Available presets: 'preset_de2', 'preset_ee2', 'preset_bacco', 'preset_aemulus', 'preset_mtu', 'preset_goku', 'preset_csstemu', 'preset_aletheia', 'preset_franken', 'preset_pkann', 'preset_hmcode'.")
            return {}

        # Single preset: just forward to get_preset_range_dict
        joint = self.get_preset_range_dict(name1)
        if name2 is None:
            return joint

        r2 = self.get_preset_range_dict(name2)
        # Union of parameter keys appearing in either preset
        keys = set(joint.keys()) | set(r2.keys())
        out = {}

        for k in keys:
            lo_list = []
            hi_list = []

            # preset 1 contribution
            if k in joint:
                lo_list.append(joint[k][0])
                hi_list.append(joint[k][1])

            # preset 2 contribution
            if k in r2:
                lo_list.append(r2[k][0])
                hi_list.append(r2[k][1])

            # If neither preset defines this key, skip (should not normally happen)
            if not lo_list or not hi_list:
                continue

            lo = max(lo_list)
            hi = min(hi_list)
            out[k] = [lo, hi]

        return out

    def check_planck_sigma(self, _p):
        x = np.array([_p["omega_b"], _p["omega_cdm"], _p["ns"]])
        # From Aletheia:
        means = np.array([0.02238396, 0.12010761, 0.96469511])
        eigenvecs = np.array([[0.99829025, -0.05633887, 0.01557255], [0.05841602, 0.97090165, -0.23224459],
                              [-0.00203502, 0.2327572, 0.97253275]])
        sigmas = np.array([0.00012303, 0.00092142, 0.00448244])

        centered = x - np.asarray(means, dtype=float)
        projected = np.asarray(eigenvecs, dtype=float).T @ centered
        deviations = np.abs(projected) / np.asarray(sigmas, dtype=float)
        return np.max(deviations)
