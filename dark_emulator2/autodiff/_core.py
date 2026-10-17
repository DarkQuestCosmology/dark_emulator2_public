import numpy as np

from .. import constants as const
from . import _setup as su

_KEYS = ("Omega_m", "omega_b", "ns", "Mnu", "Omega_k", "w0", "wa", "h0")


def _key(a):
    return None if a is None else np.asarray(a, dtype=np.float64).tobytes()


class _Emulator:
    """
    Backend-independent part of the differentiable emulator.

    Subclasses provide ``xp`` (array API namespace), ``device``, ``dtype``,
    ``_asarray`` (numpy -> backend array), ``_param`` (user value -> backend array)
    and the networks ``_amp``, ``_lin``, ``_ratio`` and ``_nlin``.
    """

    def __init__(self, de=None):
        if de is None:
            from ..dq2_emulator import DarkEmulator2
            de = DarkEmulator2()
        self.de = de
        self._cache = {}

    def _consts(self, key, build):
        if key not in self._cache:
            self._cache[key] = {k: self._asarray(v) if isinstance(v, np.ndarray) else v for k, v in build().items()}
        return self._cache[key]

    ### parameters ###

    def _amp_out(self, p, A, flag):
        xp = self.xp
        cols = xp.broadcast_arrays(p["Omega_m"], p["omega_b"], A, p["ns"], p["Mnu"], p["Omega_k"], p["w0"], p["wa"], p["h0"])
        return self._amp(xp.stack([xp.full_like(cols[0], flag)] + list(cols), axis=-1))

    def _theta(self, param):
        xp = self.xp
        p = {k: self._param(param[k]) for k in _KEYS}
        if "sigma8" in param.keys() and param["sigma8"] is not None:
            p["sigma8"] = self._param(param["sigma8"])
        else:
            if "ln(10^10As)" in param.keys() and param["ln(10^10As)"] is not None:
                lnAs = self._param(param["ln(10^10As)"])
            else:
                lnAs = xp.log(1.0e+10 * self._param(param["As"]))
            p["sigma8"] = self._amp_out(p, lnAs, 0)[..., 0] * self.de.lin_pk_emu.norm_sig8
        keys = _KEYS + ("sigma8",)
        return dict(zip(keys, xp.broadcast_arrays(*[p[k] for k in keys])))

    def get_sigma8(self, param):
        """
        sigma8 from ``As`` or ``ln(10^10As)`` (a ``sigma8`` entry is ignored).

        Parameters
        ----------
        param : dict
            Cosmological parameters (scalars or arrays with common batch shape).

        Returns
        -------
        array
            sigma8 with the batch shape of the parameters.
        """
        p = dict(param)
        p.pop("sigma8", None)
        return self._theta(p)["sigma8"]

    def get_lnAs(self, param):
        """
        ln(10^10 As) from ``sigma8``.

        Parameters
        ----------
        param : dict
            Cosmological parameters including ``sigma8``.

        Returns
        -------
        array
            ln(10^10 As) with the batch shape of the parameters.
        """
        p = self._theta(param)
        return self._amp_out(p, p["sigma8"], 1)[..., 1] * self.de.lin_pk_emu.norm_lnAs

    ### power spectra ###

    def _lin_in(self, t, z):
        xp = self.xp
        cols = [xp.expand_dims(t[k], axis=-1) for k in ("Omega_m", "omega_b", "sigma8", "ns", "Mnu", "Omega_k", "w0", "wa", "h0")]
        cols = xp.broadcast_arrays(z, xp.ones_like(z), *cols)
        return xp.stack(cols, axis=-1)

    def _reduce(self, pk, c):
        # same tail replacement as PkEmulator.reduce_shotnoise_pshot / _grad
        xp = self.xp
        logk = c["logk"]
        n = logk.shape[0]
        idx = xp.arange(n, device=self.device)
        logpk = xp.log10(xp.abs(pk))
        if c["type"] == "pshot":
            hit = logpk < c["thr"]
        else:
            hit = (logpk @ c["D_T"] >= c["thr"]) & (idx >= c["ign"])
        cut = xp.min(xp.where(hit, idx, n), axis=-1)
        cut = xp.where(cut > c["smax"], c["smax"] + 1, cut)[..., None]

        logkb = xp.broadcast_to(logk, logpk.shape)
        x1, x2 = xp.take_along_axis(logkb, cut - 1, axis=-1), xp.take_along_axis(logkb, cut - 2, axis=-1)
        y1, y2 = xp.take_along_axis(logpk, cut - 1, axis=-1), xp.take_along_axis(logpk, cut - 2, axis=-1)
        s = c["smin"]
        hi = y1 + (y1 - y2) / (x1 - x2) * (logk - x1)
        lo = logpk[..., s:s + 1] + (logpk[..., s + 1:s + 2] - logpk[..., s:s + 1]) / (logk[s + 1] - logk[s]) * (logk - logk[s])
        return 10.0**xp.where(idx >= cut, hi, xp.where(idx < s, lo, logpk))

    def _zred(self, zred):
        z = np.atleast_1d(np.asarray(zred, dtype=np.float64))
        return z, np.ndim(zred) == 0

    def _lin_pk(self, t, z, klist, pk_type, reduce_method):
        c = self._consts(("lin", z.tobytes(), _key(klist)), lambda: su.lin_consts(self.de, z, klist))
        x = self._lin_in(t, self._asarray(z))
        pk = 10.0**self._lin(x) * c["fid"]
        if pk_type == "total":
            pk = pk * self._ratio(x)
        if klist is not None:
            pk = pk @ c["out_T"]
        if reduce_method:
            k = self.de.lin_pk_emu.klist if klist is None else klist
            pk = self._reduce(pk, self._consts(("reduce", _key(k), reduce_method["type"]), lambda: su.reduce_consts(k, reduce_method)))
        return pk

    def _pk(self, t, z, klist, pk_type, reduce_method):
        xp = self.xp
        c = self._consts(("nlin", z.tobytes(), _key(klist)), lambda: su.nlin_consts(self.de, z, klist))
        zz = self._asarray(z)
        x = self._lin_in(t, zz)
        lin = 10.0**self._lin(x) * c["fid_lin"]

        sup = lin @ c["sup_T"] / c["norm_sup"]
        if c["use_log"]:
            sup = xp.log10(sup)
        cols = [xp.expand_dims(t[k], axis=-1) for k in ("Omega_m", "sigma8", "omega_b", "ns", "Mnu", "Omega_k", "w0", "wa", "h0")]
        head = xp.stack(xp.broadcast_arrays(c["head"][0] * xp.ones_like(zz), c["head"][1] * xp.ones_like(zz), zz,
                                            *(cols if c["param9d"] else [])), axis=-1)
        head = xp.broadcast_to(head, sup.shape[:-1] + head.shape[-1:])
        parts = [head, sup]
        if c["grf"] is not None:
            parts.append(xp.broadcast_to(c["grf"], sup.shape))
        out = self._nlin(xp.concat(parts, axis=-1))

        pk = (10.0**out if c["use_log"] else out) * c["norm_nl"]
        pk = c["W"] * pk + (1 - c["W"]) * (lin @ c["l2n_T"])
        if pk_type == "total":
            pk = pk * (self._ratio(x) @ c["l2n_T"])
        if klist is not None:
            pk = pk @ c["out_T"]
        if reduce_method:
            k = self.de.pk_emu.klist if klist is None else klist
            pk = self._reduce(pk, self._consts(("reduce", _key(k), reduce_method["type"]), lambda: su.reduce_consts(k, reduce_method)))
        return pk

    def get_lin_pk(self, param, zred=0.0, klist=None, pk_type="cb"):
        """
        Linear power spectrum, differentiable with respect to the parameters.

        Same conventions as :meth:`DarkEmulator2.get_lin_pk` (emulator method).

        Parameters
        ----------
        param : dict
            Cosmological parameters (scalars or arrays with common batch shape).
        zred : float or array-like, optional
            Redshift(s).
        klist : array-like, optional
            Wavenumbers [h/Mpc] to interpolate P(k) onto.
        pk_type : str, optional
            "cb" or "total".

        Returns
        -------
        k : array
            Wavenumbers [h/Mpc].
        Pk : array
            P(k) [(Mpc/h)^3] with shape (batch..., nz, nk); the z axis is dropped for scalar zred.
        """
        z, scalar = self._zred(zred)
        reduce_method = self.de.reduce_shotnoise and self.de.reduce_shotnoise_method_grad
        pk = self._lin_pk(self._theta(param), z, klist, pk_type, reduce_method)
        k = self.de.lin_pk_emu.klist if klist is None else klist
        return self._asarray(np.asarray(k, dtype=np.float64)), pk[..., 0, :] if scalar else pk

    def get_pk(self, param, zred=0.0, klist=None, pk_type="cb"):
        """
        Nonlinear power spectrum, differentiable with respect to the parameters.

        Same conventions as :meth:`DarkEmulator2.get_pk` (emulator method).

        Parameters
        ----------
        param : dict
            Cosmological parameters (scalars or arrays with common batch shape).
        zred : float or array-like, optional
            Redshift(s).
        klist : array-like, optional
            Wavenumbers [h/Mpc] to interpolate P(k) onto.
        pk_type : str, optional
            "cb" or "total".

        Returns
        -------
        k : array
            Wavenumbers [h/Mpc].
        Pk : array
            P(k) [(Mpc/h)^3] with shape (batch..., nz, nk); the z axis is dropped for scalar zred.
        """
        z, scalar = self._zred(zred)
        reduce_method = self.de.reduce_shotnoise and self.de.reduce_shotnoise_method
        pk = self._pk(self._theta(param), z, klist, pk_type, reduce_method)
        k = self.de.pk_emu.klist if klist is None else klist
        return self._asarray(np.asarray(k, dtype=np.float64)), pk[..., 0, :] if scalar else pk

    ### angular power spectra ###

    def _fK(self, chi, Ok, Dh):
        xp = self.xp
        u = Ok * (chi / Dh)**2
        small = xp.abs(u) < 1e-4
        s = xp.sqrt(xp.abs(xp.where(small, 1.0, u)))
        return chi * xp.where(small, 1 + u / 6 + u**2 / 120, xp.where(u > 0, xp.sinh(s), xp.sin(s)) / s)

    def _simpson(self, y, x):
        # scipy.integrate.simpson for irregular x along the last axis
        xp = self.xp
        n = y.shape[-1]
        h = x[..., 1:] - x[..., :-1]

        def basic(stop):
            h0, h1 = h[..., 0:stop:2], h[..., 1:stop + 1:2]
            hsum, hq = h0 + h1, h0 / h1
            return xp.sum(hsum / 6.0 * (y[..., 0:stop:2] * (2.0 - 1.0 / hq) + y[..., 1:stop + 1:2] * hsum * hsum / (h0 * h1)
                                         + y[..., 2:stop + 2:2] * (2.0 - hq)), axis=-1)

        if n % 2 == 1:
            return basic(n - 2)
        if n == 2:
            return 0.5 * h[..., 0] * (y[..., 0] + y[..., 1])
        h0, h1 = h[..., -2], h[..., -1]
        alpha = (2 * h1**2 + 3 * h0 * h1) / (6 * (h1 + h0))
        beta = (h1**2 + 3 * h0 * h1) / (6 * h0)
        eta = h1**3 / (6 * h0 * (h0 + h1))
        return basic(n - 3) + alpha * y[..., -1] + beta * y[..., -2] - eta * y[..., -3]

    def _limber(self, t, pk, c):
        # same as derived.lensing.pk2cl_limber_src_dist; pk in (Mpc/h)^3 on c["k"] [h/Mpc], shape (..., nz, nk)
        xp = self.xp
        h, Om, Ok, w0, wa = (xp.expand_dims(t[k], axis=-1) for k in ("h0", "Omega_m", "Omega_k", "w0", "wa"))
        zq, z = c["zq"], c["z"]
        Fde = (1 + zq)**(3 * (1 + w0 + wa)) * xp.exp(-3 * wa * zq / (1 + zq))
        invE = 1.0 / xp.sqrt(Om * (1 + zq)**3 + Ok * (1 + zq)**2 + (1.0 - Om - Ok) * Fde)
        Dh = const.cspeed / (100 * h)  # [Mpc]
        chi = Dh * (invE @ c["cum_T"])
        fk = self._fK(chi, Ok, Dh)
        pos = fk > 0
        fk_safe = xp.where(pos, fk, 1.0)

        norm = 1.5 * Om * (100 * h / const.cspeed)**2
        if "nw" in c:
            g = self._fK(chi[..., None, :] - chi[..., :, None], Ok[..., None], Dh[..., None]) / fk_safe[..., None, :]
            W = norm * (1 + z) * fk * xp.sum(xp.where(c["upper"] > 0, g, 0.0) * c["nw"], axis=-1)
        else:
            W = norm * (1 + z) * self._fK(chi[..., -1:] - chi, Ok, Dh) * fk / fk[..., -1:]
        W2 = xp.where(pos, (W / fk_safe)**2, 0.0)

        k = c["k"]
        q = (c["l"] + 0.5) / (fk_safe[..., None] * h[..., None])  # (..., nz, nl) [h/Mpc]
        j = xp.clip(xp.searchsorted(k, q), 1, k.shape[0] - 1)
        kb = xp.broadcast_to(k, pk.shape)
        k0, k1 = xp.take_along_axis(kb, j - 1, axis=-1), xp.take_along_axis(kb, j, axis=-1)
        p0, p1 = xp.take_along_axis(pk, j - 1, axis=-1), xp.take_along_axis(pk, j, axis=-1)
        pq = xp.where((q >= k[0]) & (q <= k[-1]), p0 + (q - k0) / (k1 - k0) * (p1 - p0), 0.0) / h[..., None]**3
        return self._simpson(xp.matrix_transpose(W2[..., None] * pq), chi[..., None, :])

    def _cl(self, param, zlist, klist, llist, src_dist, get):
        t = self._theta(param)
        c = self._consts(("limber", _key(zlist), _key(klist), _key(llist), _key(src_dist)),
                         lambda: su.limber_consts(zlist, klist, llist, src_dist))
        return self._asarray(np.asarray(llist, dtype=np.float64)), self._limber(t, get(t), c)

    def get_lin_cl_limber(self, param, zlist=np.linspace(0, 3, 151), klist=np.logspace(-3, 3, 601), llist=np.logspace(0.0, 5.0, 601),
                          pk_type="total", src_dist=None):
        """
        Angular power spectrum in the Limber approximation with the linear P(k).

        Same conventions as :meth:`DarkEmulator2.get_lin_cl_limber` (emulator method).

        Returns
        -------
        l : array
            Multipoles.
        Cl : array
            C(l) with shape (batch..., nl).
        """
        z = np.asarray(zlist, dtype=np.float64)
        return self._cl(param, z, klist, llist, src_dist,
                        lambda t: self._lin_pk(t, z, np.asarray(klist, dtype=np.float64), pk_type, self.de.reduce_shotnoise_method_grad))

    def get_cl_limber(self, param, zlist=np.linspace(0, 3, 151), klist=np.logspace(-3, 3, 601), llist=np.logspace(0.0, 5.0, 601),
                      pk_type="total", src_dist=None):
        """
        Angular power spectrum in the Limber approximation with the nonlinear P(k).

        Same conventions as :meth:`DarkEmulator2.get_cl_limber` (emulator method).

        Returns
        -------
        l : array
            Multipoles.
        Cl : array
            C(l) with shape (batch..., nl).
        """
        z = np.asarray(zlist, dtype=np.float64)
        return self._cl(param, z, klist, llist, src_dist,
                        lambda t: self._pk(t, z, np.asarray(klist, dtype=np.float64), pk_type, self.de.reduce_shotnoise_method))
