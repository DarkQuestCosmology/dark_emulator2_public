from .__version__ import __version__

import numpy as np
import torch
from scipy.interpolate import InterpolatedUnivariateSpline as ius

from .parameters import Parameters
from .reg_lin_pk import LinPkEmulator
from .reg_pk import PkEmulator

from . import cosm_tools as ct
from . import derived as dv


class DarkEmulator2:
    def __init__(self, npart=3000, lbox=1000,
                 verbose=False, debug=False):
        """
        DarkEmulator2 class

        Provides cosmological observables including the power spectrum P(k),
        correlation function xi(r), and angular power spectrum C(ℓ),
        computed from linear and non-linear models.

        Supports emulation and CLASS-based calculations.

        Parameters
        ----------
        npart : int, optional
            Number of particles per dimension (default: 3000).
        lbox : float, optional
            Box size in [Mpc/h] (default: 1000).
        verbose : bool, optional
            Verbose mode (default: False).
        """

        self.__version__ = __version__

        self.npart = npart
        self.lbox = lbox

        self.param = Parameters()
        self.lin_pk_emu = LinPkEmulator()
        self.pk_emu = PkEmulator(npart, lbox, debug)

        self.lin_pk_switch = True
        self.lin_pk_switch_k = 1.0e-2
        self.nsmooth = 20

        self.reduce_shotnoise = False
        self.reduce_shotnoise_method_grad = {"type": "grad", "base": self.kny, "factor": -1.5}
        self.reduce_shotnoise_method_pshot = {"type": "pshot", "base": self.pshot, "factor": 10.0}
        self.reduce_shotnoise_method = self.reduce_shotnoise_method_pshot

        self.verbose = verbose
        self.output_verbose(verbose)
        self.show_version(verbose)
        self.show_mode(verbose)

    def show_version(self, flag=True):
        if flag:
            print("### Dark Emulator2 version :: ", self.__version__)
            print("### Pytorch version :: ", torch.__version__)

    def show_mode(self, flag=True):
        if flag:
            print(f"npart(1D) :: {self.npart}")
            print(f"lbox :: {self.lbox} [Mpc/h]")
            print(f"Nyquist frequency is {self.kny:.4f} [h/Mpc].")
            print(f"Shotnoise limit is {self.pshot:.4f}  [(Mpc/h)^3].")

            print(f"lin_pk_switch :: {self.lin_pk_switch}")
            if self.lin_pk_switch:
                print(f"lin_pk_switch_k :: {self.lin_pk_switch_k} [h/Mpc]")
                print(f"pk-nsmooth :: {self.nsmooth}")
            print(f"reduce_shotnoise :: {self.reduce_shotnoise}")
            if self.reduce_shotnoise:
                print(f"reduce_shotnoise_method :: {self.reduce_shotnoise_method}")

    def output_verbose(self, flag=True):
        self.verbose = flag
        self.param.verbose = flag
        self.lin_pk_emu.verbose = flag
        self.pk_emu.verbose = flag

    @property
    def kny(self):
        """
        Nyquist frequency [h/Mpc].

        Computed as:
            π (npart / lbox)

        Returns
        -------
        float
            Nyquist frequency in [h/Mpc].
        """

        return ct.Nyquist_freq(self.npart, self.lbox)

    @property
    def pshot(self):
        """
        Shot noise level [(Mpc/h)^3].

        Computed as:
            (lbox^3) / (npart^3)

        Returns
        -------
        float
            Shot noise in [(Mpc/h)^3].
        """
        return ct.Pshot(self.npart, self.lbox)

    def __sorting_zred(self, zred):
        zred = np.asarray(zred)
        assert np.all(zred <= 5), "Input redshift is too large, specify z<5. Have you entered klist in zred by mistake?"

        scalar_input = False
        reverse_zred = False
        if zred.ndim == 0:
            zred = zred[np.newaxis]
            scalar_input = True

        if not scalar_input and zred.size > 1:
            if np.diff(zred)[0] < 0:
                zred = np.sort(zred)
                reverse_zred = True
        return zred, scalar_input, reverse_zred

    ### for amplitude parameter ###

    def get_sigma8(self, param, method="emulator"):
        """
        Compute sigma8 from input cosmological parameters.

        This updates the input parameter dictionary in-place
        by setting param["sigma8"].

        Parameters
        ----------
        param : dict
            Cosmological parameter dictionary.
        method : str, optional
            "emulator" (default) or "class".

        Returns
        -------
        float
            Computed sigma8.
        """

        self.param.clear_sigma8(param)
        if method == "emulator":
            sig8 = self.lin_pk_emu.get_sigma8(param)
        elif method == "class":
            sig8 = self.lin_pk_emu.get_class_sigma8(param)

        # overwrite sigma8 of input parameter
        param["sigma8"] = sig8
        return sig8

    def get_As(self, param):
        """
        Compute primordial amplitude As from sigma8.

        This updates the input parameter dictionary in-place
        by setting param["As"].

        Parameters
        ----------
        param : dict
            Cosmological parameter dictionary. Must have "sigma8".

        Returns
        -------
        float
            Computed As.
        """

        assert param["sigma8"] is not None
        self.param.clear_As(param)
        As = self.lin_pk_emu.get_As(param)
        # overwrite As of input parameter
        param["As"] = As
        return As

    def get_lnAs(self, param):
        """
        Compute primordial amplitude ln(10^10 As) from sigma8.

        This updates the input parameter dictionary in-place
        by setting param["ln(10^10As)"].

        Parameters
        ----------
        param : dict
            Cosmological parameter dictionary. Must have "sigma8".

        Returns
        -------
        float
            Computed ln(10^10 As).
        """

        # Here, "lnAs" is "ln(10^10As)"
        assert param["sigma8"] is not None
        self.param.clear_As(param)
        lnAs = self.lin_pk_emu.get_lnAs(param)
        # overwrite As of input parameter
        param["ln(10^10As)"] = lnAs
        return lnAs

    ### for linear power spectrum ###
    def get_lin_pk(self, param, zred=0.0, klist=None, method="emulator", pk_type="cb"):
        """
        Compute the linear power spectrum P(k).

        Parameters
        ----------
        param : dict
            Cosmological parameter dictionary.
        zred : float or array-like, optional
            Redshift(s) at which to compute P(k). Default is 0.0.
        klist : array-like, optional
            Wavenumbers [h/Mpc] to interpolate P(k) onto.
        method : str, optional
            "emulator" (default) or "class".
        pk_type : str, optional
            "cb" for CDM+baryon or "total" for total matter.

        Returns
        -------
        k : array
            Wavenumber array [h/Mpc].
        Pk : array
            Linear power spectrum P(k) [(Mpc/h)^3].
        """

        zred, scalar_input, reverse_zred = self.__sorting_zred(zred)
        if method == "emulator":
            lin_pk = self.lin_pk_emu.get_lin_pk(param, zred, pk_type)
            k = self.lin_pk_emu.klist
        elif method == "class":
            lin_pk = self.lin_pk_emu.get_class_lin_pk(param, zred, pk_type)
            k = self.lin_pk_emu.class_klist

        if klist is not None:
            lin_pk = np.array([ius(np.log(k), _pk)(np.log(klist)) for _pk in lin_pk])
            k = klist

        # Linear pk has no shot noise but is for extensions up to k>100.
        if self.reduce_shotnoise:
            reduce_method = self.reduce_shotnoise_method_grad
            lin_pk = self.pk_emu.reduce_shotnoise(lin_pk, k, reduce_method)
            if self.verbose:
                print("Remove shot noise artificially by extrapolation.")

        if scalar_input:
            lin_pk = np.squeeze(lin_pk)

        if reverse_zred:
            # Here, not scalar_input and reversing axis=1
            lin_pk = lin_pk[::-1]

        return k, lin_pk

    ### alias of linear power spectrum ###

    def get_lin_pk_cb(self, param, zred=0.0, klist=None, method="emulator"):
        """
        Alias for :meth:`get_lin_pk` with ``pk_type="cb"``.
        See :meth:`get_lin_pk` for details.
        """

        return self.get_lin_pk(param, zred, klist, method, pk_type="cb")

    def get_lin_pk_total(self, param, zred=0.0, klist=None, method="emulator"):
        """
        Alias for :meth:`get_lin_pk` with ``pk_type="total"``.
        See :meth:`get_lin_pk` for details.
        """

        return self.get_lin_pk(param, zred, klist, method, pk_type="total")

    def get_lin_pk_ratio(self, param, zred=0.0, klist=None, method="emulator"):
        """
        Compute the ratio of total matter to CDM+baryon linear power spectra.

        Parameters
        ----------
        param : dict
            Cosmological parameter dictionary.
        zred : float or array-like, optional
            Redshift(s) at which to compute ratio. Default is 0.0.
        klist : array-like, optional
            List of wavenumbers [h/Mpc] to interpolate the result onto.
        method : str, optional
            "emulator" (default) or "class".

        Returns
        -------
        k : array
            Wavenumber array [h/Mpc].
        ratio : array
            Ratio P_total(k) / P_cb(k).
        """

        zred, scalar_input, reverse_zred = self.__sorting_zred(zred)
        k, lin_ratio = self.lin_pk_emu.get_ratio_of_cb_to_total(param, zred, method=method)

        if klist is not None:
            lin_ratio = np.array([ius(k, _ratio)(klist) for _ratio in lin_ratio])
            k = klist

        if scalar_input:
            lin_ratio = np.squeeze(lin_ratio)

        if reverse_zred:
            # Here, not scalar_input and reversing axis=1
            lin_ratio = lin_ratio[::-1]

        return k, lin_ratio

    ### for non-linear power spectrum ###

    def get_pk(self, param, zred=0.0, klist=None, method="emulator", pk_type="cb"):
        """
        Compute the non-linear power spectrum P(k).

        Parameters
        ----------
        param : dict
            Cosmological parameter dictionary.
        zred : float or array-like, optional
            Redshift(s) at which to compute P(k). Default is 0.0.
        klist : array-like, optional
            Wavenumbers [h/Mpc] to interpolate P(k) onto.
        method : str, optional
            "emulator" (default) or "class". The nonlinear emulator is always
            used; "class" uses CLASS for the linear power spectrum and sigma8
            inputs to the nonlinear emulator.
        pk_type : str, optional
            "cb" or "total".

        Returns
        -------
        k : array
            Wavenumber array [h/Mpc].
        Pk : array
            Non-linear power spectrum P(k) [(Mpc/h)^3].
        """

        zred, scalar_input, reverse_zred = self.__sorting_zred(zred)
        pk = self.pk_emu.get_pk(param, zred, method)
        k = self.pk_emu.klist

        if self.lin_pk_switch:
            _, lin_pk = self.get_lin_pk_cb(param, klist=k, zred=np.squeeze(zred), method=method)
            kidx = np.abs(k - self.lin_pk_switch_k).argmin()
            w = np.linspace(0.0, 1.0, self.nsmooth, endpoint=False)

            if lin_pk.ndim > 1:
                pk[:, k <= self.lin_pk_switch_k] = lin_pk[:, k <= self.lin_pk_switch_k]
                for i in range(len(pk)):
                    pk[i][kidx:kidx + self.nsmooth] = w * pk[i][kidx:kidx + self.nsmooth] + (1 - w) * lin_pk[i][kidx:kidx + self.nsmooth]
            else:
                pk[:, k <= self.lin_pk_switch_k] = lin_pk[k <= self.lin_pk_switch_k]
                for i in range(len(pk)):
                    pk[i][kidx:kidx + self.nsmooth] = w * pk[i][kidx:kidx + self.nsmooth] + (1 - w) * lin_pk[kidx:kidx + self.nsmooth]

        if pk_type == "total":
            # pk_tot = pk_cb / (lin_pk_cb / lin_pk_total)
            k2, lin_ratio = self.lin_pk_emu.get_ratio_of_cb_to_total(param, zred, method=method)
            lin_ratio = np.array([ius(np.log(k2), _pkr)(np.log(k)) for _pkr in lin_ratio])
            pk *= lin_ratio

        if klist is not None:
            pk = np.array([ius(np.log(k), _pk)(np.log(klist)) for _pk in pk])
            k = klist

        if self.reduce_shotnoise:
            pk = self.pk_emu.reduce_shotnoise(pk, k, self.reduce_shotnoise_method)
            if self.verbose:
                print("Remove shot noise artificially by extrapolation.")

        if scalar_input:
            pk = np.squeeze(pk)

        if reverse_zred:
            # Here, not scalar_input and reversing axis=1
            pk = pk[::-1]

        return k, pk

    ### alias of non-linear power spectrum ###

    def get_pk_cb(self, param, zred=0.0, klist=None, method="emulator"):
        """
        Alias for :meth:`get_pk` with ``pk_type="cb"``.
        See :meth:`get_pk` for details.
        """

        return self.get_pk(param, zred, klist, method, pk_type="cb")

    def get_pk_total(self, param, zred=0.0, klist=None, method="emulator"):
        """
        Alias for :meth:`get_pk` with ``pk_type="total"``.
        See :meth:`get_pk` for details.
        """

        return self.get_pk(param, zred, klist, method, pk_type="total")

    ### for correlation function ###

    def get_lin_xi(self, param, zred=0.0, method="emulator", pk_type="total"):
        """
        Compute the linear real-space correlation function xi(r)
        by Hankel transform of the linear power spectrum P(k).

        Parameters
        ----------
        param : dict
            Cosmological parameter dictionary.
        zred : float, optional
            Redshift at which to compute xi(r). Default is 0.0.
        method : str, optional
            "emulator" (default) or "class".
        pk_type : str, optional
            "total" or "cb".

        Returns
        -------
        r : array
            Separation distances [Mpc/h].
        xi : array
            Linear correlation function xi(r).
        """

        klist = np.logspace(-3, 3, 601)
        N_extrap_low = 0
        N_extrap_high = 10000

        rshot_flag = self.reduce_shotnoise
        self.reduce_shotnoise = True
        k, lin_pk = self.get_lin_pk(param, zred=zred, klist=klist, method=method, pk_type=pk_type)
        if lin_pk.ndim > 1:
            xi = [[] for _ in range(len(lin_pk))]
            for i in range(len(lin_pk)):
                r, xi[i] = dv.pk2xi(k, lin_pk[i], N_extrap_low=N_extrap_low, N_extrap_high=N_extrap_high)
        else:
            r, xi = dv.pk2xi(k, lin_pk, N_extrap_low=N_extrap_low, N_extrap_high=N_extrap_high)

        self.reduce_shotnoise = rshot_flag
        return r, np.array(xi)

    def get_xi(self, param, zred=0.0, method="emulator", pk_type="total"):
        """
        Compute the non-linear real-space correlation function xi(r)
        by Hankel transform of the non-linear power spectrum P(k).

        Parameters
        ----------
        param : dict
            Cosmological parameter dictionary.
        zred : float, optional
            Redshift at which to compute xi(r). Default is 0.0.
        method : str, optional
            "emulator" (default) or "class". The nonlinear emulator is always
            used; "class" uses CLASS for the linear power spectrum and sigma8
            inputs to the nonlinear emulator.
        pk_type : str, optional
            "total" or "cb".

        Returns
        -------
        r : array
            Separation distances [Mpc/h].
        xi : array
            Non-linear correlation function xi(r).
        """

        klist = np.logspace(-3, 3, 601)
        N_extrap_low = 0
        N_extrap_high = 10000

        rshot_flag = self.reduce_shotnoise
        self.reduce_shotnoise = True
        k, pk = self.get_pk(param, zred=zred, klist=klist, method=method, pk_type=pk_type)
        if pk.ndim > 1:
            xi = [[] for _ in range(len(pk))]
            for i in range(len(pk)):
                r, xi[i] = dv.pk2xi(k, pk[i], N_extrap_low=N_extrap_low, N_extrap_high=N_extrap_high)
        else:
            r, xi = dv.pk2xi(k, pk, N_extrap_low=N_extrap_low, N_extrap_high=N_extrap_high)

        self.reduce_shotnoise = rshot_flag
        return r, np.array(xi)

    ### for cosmic shear ###
    def get_lin_cl_limber(self, param,
                          zlist=np.linspace(0, 3, 151), klist=np.logspace(-3, 3, 601), llist=np.logspace(0.0, 5.0, 601),
                          method="emulator", pk_type="total", src_dist=None):
        """
        Compute the angular power spectrum C(l) using Limber approximation
        with the linear P(k).

        Parameters
        ----------
        param : dict
            Cosmological parameter dictionary.
        zlist : array-like
            Redshift sampling for integration.
        klist : array-like
            k sampling for P(k).
        llist : array-like
            Multipoles ell.
        method : str, optional
            "emulator" (default) or "class".
        pk_type : str, optional
            "total" or "cb".
        src_dist : array-like, optional
            Source galaxy redshift distribution normalized to 1.

        Returns
        -------
        l : array
            Multipole moments.
        Cl : array
            Angular power spectrum C(l).
        """

        assert not np.isscalar(zlist)
        if src_dist is not None:
            src_dist = np.asarray(src_dist)
            assert src_dist.ndim == 1
        rshot_flag = self.reduce_shotnoise
        self.reduce_shotnoise = True

        if self.verbose:
            print(f"param: {param}")
            print(f"zlist[{len(zlist)}]: {zlist[0]:.3f} -> {zlist[-1]:.3f}, "
                  f"klist[{len(klist)}]: {klist[0]:.1e} -> {klist[-1]:.1e}, "
                  f"llist[{len(llist)}]: {llist[0]:.1f} -> {llist[-1]:.1f}")
            print(f"method={method}, pk_type={pk_type}", end=", ")
            if src_dist is None:
                print(f"src_dist=delta(z={zlist[-1]:.3f})")
            else:
                print(f"src_dist=shape{np.shape(src_dist)} (normalized)")

        _k, _pk = self.get_lin_pk(param, klist=klist, zred=zlist, method=method, pk_type=pk_type)
        l, cl = dv.pk2cl_limber_src_dist(param, pk=_pk, klist=_k, zlist=zlist, llist=llist, nsz=src_dist, modified=True)
        self.reduce_shotnoise = rshot_flag
        return l, cl

    def get_cl_limber(self, param,
                      zlist=np.linspace(0, 3, 151), klist=np.logspace(-3, 3, 601), llist=np.logspace(0.0, 5.0, 601),
                      method="emulator", pk_type="total", src_dist=None):
        """
        Compute the angular power spectrum C(l) using Limber approximation
        with the non-linear power spectrum P(k).

        Parameters
        ----------
        param : dict
            Cosmological parameter dictionary.
        zlist : array-like
            Redshift sampling for integration.
        klist : array-like
            k sampling for P(k).
        llist : array-like
            Multipoles ell.
        method : str, optional
            "emulator" (default) or "class". The nonlinear emulator is always
            used; "class" uses CLASS for the linear power spectrum and sigma8
            inputs to the nonlinear emulator.
        pk_type : str, optional
            "total" or "cb".
        src_dist : array-like, optional
            Source galaxy redshift distribution normalized to 1.

        Returns
        -------
        l : array
            Multipole moments.
        Cl : array
            Angular power spectrum C(l).
        """

        assert not np.isscalar(zlist)
        rshot_flag = self.reduce_shotnoise
        self.reduce_shotnoise = True

        if self.verbose:
            print(f"param: {param}")
            print(f"zlist[{len(zlist)}]: {zlist[0]:.3f} -> {zlist[-1]:.3f}, "
                  f"klist[{len(klist)}]: {klist[0]:.1e} -> {klist[-1]:.1e}, "
                  f"llist[{len(llist)}]: {llist[0]:.1f} -> {llist[-1]:.1f}")
            print(f"method={method}, pk_type={pk_type}", end=", ")
            if src_dist is None:
                print(f"src_dist=delta(z={zlist[-1]:.3f})")
            else:
                print(f"src_dist=shape{np.shape(src_dist)} (normalized)")

        _k, _pk = self.get_pk(param, klist=klist, zred=zlist, method=method, pk_type=pk_type)
        l, cl = dv.pk2cl_limber_src_dist(param, pk=_pk, klist=_k, zlist=zlist, llist=llist, nsz=src_dist, modified=True)
        self.reduce_shotnoise = rshot_flag
        return l, cl

    ### for utils ###
    def get_power_distance(self, param, method="emulator"):
        """
        Calculate power distance metric as a diagnostic of the linear power spectrum.

        Parameters
        ----------
        param : dict
            Cosmological parameter dictionary.
        method : str, optional
            "emulator" (default) or "class".

        Returns
        -------
        float
            Power distance metric.
        """

        zred = 0.0
        zred = self.__sorting_zred(zred)[0]
        pk_type = "cb"
        logk_min = -3.0
        logk_max = 2.0
        # logk_max = 1.0
        kbin = 1001
        klist = np.logspace(logk_min, logk_max, kbin)

        if method == "emulator":
            lin_pk = self.lin_pk_emu.get_lin_pk(param, zred, pk_type)
            k = self.lin_pk_emu.klist
        elif method == "class":
            lin_pk = self.lin_pk_emu.get_class_lin_pk(param, zred, pk_type)
            k = self.lin_pk_emu.class_klist

        lin_pk_spl = ius(np.log(k), lin_pk[0])
        lin_pk = lin_pk_spl(np.log(klist))
        k = klist

        power_dist = self.lin_pk_emu.calc_power_distance(k, lin_pk)
        return power_dist

    def get_pk_cb_noiselist(self, param, zred=0.0, klist=None, method="emulator", noise_list=None):
        # for validation
        zred, scalar_input, reverse_zred = self.__sorting_zred(zred)
        pk = self.pk_emu.get_pk(param, zred, method, noise_list)
        k = self.pk_emu.klist

        if self.lin_pk_switch:
            _, lin_pk = self.get_lin_pk_cb(param, klist=k, zred=np.squeeze(zred), method=method)
            kidx = np.abs(k - self.lin_pk_switch_k).argmin()
            w = np.linspace(0.0, 1.0, self.nsmooth, endpoint=False)

            if lin_pk.ndim > 1:
                pk[:, k <= self.lin_pk_switch_k] = lin_pk[:, k <= self.lin_pk_switch_k]
                for i in range(len(pk)):
                    pk[i][kidx:kidx + self.nsmooth] = w * pk[i][kidx:kidx + self.nsmooth] + (1 - w) * lin_pk[i][kidx:kidx + self.nsmooth]
            else:
                pk[:, k <= self.lin_pk_switch_k] = lin_pk[k <= self.lin_pk_switch_k]
                for i in range(len(pk)):
                    pk[i][kidx:kidx + self.nsmooth] = w * pk[i][kidx:kidx + self.nsmooth] + (1 - w) * lin_pk[kidx:kidx + self.nsmooth]

        if klist is not None:
            pk = np.array([ius(np.log(k), _pk)(np.log(klist)) for _pk in pk])
            k = klist

        if scalar_input:
            pk = np.squeeze(pk)

        if reverse_zred:
            # Here, not scalar_input and reversing axis=1
            pk = pk[::-1]
        return k, pk
