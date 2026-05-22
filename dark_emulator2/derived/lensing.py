import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import RegularGridInterpolator as rgi
from .. import constants as const


def comoving_distance(z, p):
    """
    Comoving distance chi(z).

    Parameters
    ----------
    z : float or ndarray
        Redshift(s).
    p : dict
        Cosmological parameters with keys "h0", "Omega_m", "Omega_de", "Omega_k".

    Returns
    -------
    cdist : float or ndarray
        Comoving distance [Mpc].
    """

    H0, Om, Ode, Ok = 100 * p["h0"], p["Omega_m"], p["Omega_de"], p["Omega_k"]

    # H0 [km/s/Mpc]
    Dh = const.cspeed / H0  # Hubble-Distance [Mpc/h]

    def inv_E(z): return 1.0 / (np.sqrt(Om * (1 + z)**3 + Ok * (1 + z)**2 + Ode))

    if np.isscalar(z):
        integral = quad(inv_E, 0, z)[0]
        cdist = Dh * integral
    else:
        integral = np.array([quad(inv_E, 0, _z)[0] for _z in z])
        cdist = Dh * integral

    return cdist  # [Mpc]


def proper_distance(z, p):
    """
    Transverse comoving distance D_M(z).

    In flat universe D_M = χ.

    Parameters
    ----------
    z : float or ndarray
        Redshift(s).
    p : dict
        Cosmological parameters.

    Returns
    -------
    pdist : float or ndarray
        Proper distance D_M(z) [Mpc].
    """

    H0, Ok = 100 * p["h0"], p["Omega_k"]
    x = comoving_distance(z, p)
    Dh = const.cspeed / H0  # Hubble-Distance [Mpc/h]
    sqrt_Ok = np.sqrt(np.abs(Ok))

    if Ok > 0: pdist = (Dh / sqrt_Ok) * np.sinh(x * sqrt_Ok / Dh)
    elif Ok == 0: pdist = x
    elif Ok < 0: pdist = (Dh / sqrt_Ok) * np.sin(x * sqrt_Ok / Dh)
    return pdist  # [Mpc]


def proper_distance_x(x, p):
    """
    Transverse comoving distance D_M(x) for input comoving distance.

    In flat universe D_M = x.

    Parameters
    ----------
    x : float or ndarray
        Comoving distance(s).
    p : dict
        Cosmological parameters.

    Returns
    -------
    pdist : float or ndarray
        Proper distance D_M(x) [Mpc].
    """

    H0, Ok = 100 * p["h0"], p["Omega_k"]
    Dh = const.cspeed / H0  # Hubble-Distance [Mpc/h]
    sqrt_Ok = np.sqrt(np.abs(Ok))

    if Ok > 0: pdist = (Dh / sqrt_Ok) * np.sinh(x * sqrt_Ok / Dh)
    elif Ok == 0: pdist = x
    elif Ok < 0: pdist = (Dh / sqrt_Ok) * np.sin(x * sqrt_Ok / Dh)
    return pdist  # [Mpc]


def angular_distance(z, p):
    """
    Angular diameter distance D_A(z).

    D_A(z) = D_M(z) / (1+z)

    Parameters
    ----------
    z : float or ndarray
        Redshift(s).
    p : dict
        Cosmological parameters.

    Returns
    -------
    adist : float or ndarray
        Angular diameter distance D_A(z) [Mpc].
    """

    adist = proper_distance(z, p) / (1 + z)
    return adist  # [Mpc]


def weight_function(zs, z, p):
    """
    Lensing weight function W(χ) for redshifts.

    Parameters
    ----------
    zs : float
        Source redshift.
    z : float or ndarray
        Lens redshift(s).
    p : dict
        Cosmological parameters.

    Returns
    -------
    Wx : float or ndarray
        Lensing weight function [1/Mpc].
    """

    assert np.all(zs >= z)

    H0, Om = 100 * p["h0"], p["Omega_m"]
    norm = (3.0 * Om / 2.0) * (H0 / const.cspeed)**2.0
    kernel = (1.0 + z) * (proper_distance(zs, p) - proper_distance(z, p)) * proper_distance(z, p) / proper_distance(zs, p)
    Wx = norm * kernel
    return Wx


def weight_function_x(xs, x, z, p):
    """
    Lensing weight function W(x) for comoving distances.

    Parameters
    ----------
    xs : float
        Source comoving distance.
    x : float or ndarray
        Lens comoving distance(s).
    z : float or ndarray
        Lens redshift(s).
    p : dict
        Cosmological parameters.

    Returns
    -------
    Wx : float or ndarray
        Lensing weight function [1/Mpc].
    """

    assert np.all(xs >= x)

    H0, Om = 100 * p["h0"], p["Omega_m"]
    norm = (3.0 * Om / 2.0) * (H0 / const.cspeed)**2.0
    kernel = (1.0 + z) * proper_distance_x(xs - x, p) * proper_distance_x(x, p) / proper_distance_x(xs, p)
    Wx = norm * kernel
    return Wx


def lensing_kernel_x(xs, x, z, p):
    """
    Lensing kernel W(x) for comoving distances.

    Same as weight_function_x, often used with integrals over source distributions.

    Parameters
    ----------
    xs : float
        Source comoving distance.
    x : float or ndarray
        Lens comoving distance(s).
    z : float or ndarray
        Lens redshift(s).
    p : dict
        Cosmological parameters.

    Returns
    -------
    Wx : float or ndarray
        Lensing kernel [1/Mpc].
    """

    assert np.all(xs >= x)

    H0, Om = 100 * p["h0"], p["Omega_m"]
    norm = (3.0 * Om / 2.0) * (H0 / const.cspeed)**2.0
    kernel = (1.0 + z) * proper_distance_x(xs - x, p) * proper_distance_x(x, p) / proper_distance_x(xs, p)
    Wx = norm * kernel
    return Wx


def lensing_kernel(zs, nsz, ns_zlist, p):
    """
    Lensing kernel integrated over a source redshift distribution.

    Parameters
    ----------
    zs : float
        Lens redshift.
    nsz : ndarray
        Normalized source redshift distribution evaluated at ns_zlist.
    ns_zlist : ndarray
        Redshift values for nsz.
    p : dict
        Cosmological parameters.

    Returns
    -------
    Wx : float
        Integrated lensing kernel [1/Mpc].
    """

    chi = comoving_distance(zs, p)
    chi_s = comoving_distance(ns_zlist, p)
    Wx_integrand = lensing_kernel_x(chi_s, chi, zs, p) * nsz
    Wx = simpson(Wx_integrand, x=chi_s)
    return Wx


def pk2cl_limber(p, pk, klist, zlist, llist, modified=True):
    # source function is delta distribution, i.e. n(z) = delta(z - zs)
    # In this function, L=Mpc, T=s units.
    """
    Compute angular power spectrum C_ell using Limber approximation
    for a delta-function source distribution.

    This calculates
        C_ell = ∫ W(χ)^2 P(k= (l+0.5)/f_K(χ), z) dχ
    where W(χ) is the lensing weight function.

    Parameters
    ----------
    p : dict
        Cosmological parameters.
    pk : ndarray
        Power spectrum array with shape (len(zlist), len(klist)).
    klist : ndarray
        Wavenumber array [h/Mpc].
    zlist : ndarray
        Redshift array.
    llist : ndarray
        Multipole moments.
    modified : bool, optional
        If True, use (l+0.5) for Limber; otherwise use l.

    Returns
    -------
    llist : ndarray
        Multipole moments.
    Cl : ndarray
        Angular power spectrum C_ell.
    """

    assert np.diff(zlist)[0] > 0
    assert pk.shape == zlist.shape + klist.shape

    h = p["h0"]

    _klist = klist / h  # [h/Mpc] to [1/Mpc]
    _pk = pk * h**3  # [(Mpc/h)^3] to [(Mpc)^3]

    chi = comoving_distance(zlist, p)  # [Mpc]
    fKchi = proper_distance(zlist, p)  # [Mpc]

    chis = chi[-1]
    W_chi = weight_function_x(chis, chi, zlist, p)
    W_chi = (W_chi / (fKchi + 1e-20))**2

    if modified:
        chi_inv = np.outer(llist + 0.5, 1.0 / (fKchi + 1e-20))
    else:
        chi_inv = np.outer(llist, 1.0 / (fKchi + 1e-20))

    power_rgi = rgi((zlist, _klist), _pk, bounds_error=False, fill_value=0.0)
    points_2d = np.dstack([np.meshgrid(zlist, chi_inv.flatten(), indexing='ij')])
    power_integrand = power_rgi(points_2d.T).reshape(len(llist), len(zlist), len(zlist))
    power_integrand = power_integrand.diagonal(axis1=1, axis2=2)
    full_integrand = W_chi[np.newaxis, :] * power_integrand
    Cl = simpson(full_integrand, x=chi, axis=1)
    return llist, Cl


def pk2cl_limber_src_dist(p, pk, klist, zlist, llist, nsz=None, modified=True):
    # source function is arbitrary distribution, i.e. n(z) = nsz(z)
    # In this function, L=Mpc, T=s units.
    """
    Compute angular power spectrum C_ell using Limber approximation
    for an arbitrary source redshift distribution n(z).

    If nsz is None, reduces to delta-function case.

    Parameters
    ----------
    p : dict
        Cosmological parameters.
    pk : ndarray
        Power spectrum array with shape (len(zlist), len(klist)).
    klist : ndarray
        Wavenumber array [h/Mpc].
    zlist : ndarray
        Redshift array.
    llist : ndarray
        Multipole moments.
    nsz : ndarray, optional
        Normalized source redshift distribution over zlist.
    modified : bool, optional
        If True, use (l+0.5) for Limber; otherwise use l.

    Returns
    -------
    llist : ndarray
        Multipole moments.
    Cl : ndarray
        Angular power spectrum C_ell.
    """

    if nsz is None:
        return pk2cl_limber(p, pk, klist, zlist, llist, modified)

    assert np.diff(zlist)[0] > 0
    assert pk.shape == zlist.shape + klist.shape

    nsz = np.asarray(nsz)
    assert nsz.shape == zlist.shape, f"nsz.shape {nsz.shape} != zlist.shape {zlist.shape}"

    h = p["h0"]

    _klist = klist / h  # [h/Mpc] to [1/Mpc]
    _pk = pk * h**3  # [(Mpc/h)^3] to [(Mpc)^3]

    chi = comoving_distance(zlist, p)  # [Mpc]
    fKchi = proper_distance(zlist, p)  # [Mpc]

    chis = chi[-1]
    W_chi = weight_function_x(chis, chi, zlist, p) * nsz
    W_chi = (W_chi / (fKchi + 1e-20))**2

    if modified:
        chi_inv = np.outer(llist + 0.5, 1.0 / (fKchi + 1e-20))
    else:
        chi_inv = np.outer(llist, 1.0 / (fKchi + 1e-20))

    power_rgi = rgi((zlist, _klist), _pk, bounds_error=False, fill_value=0.0)
    points_2d = np.dstack([np.meshgrid(zlist, chi_inv.flatten(), indexing='ij')])
    power_integrand = power_rgi(points_2d.T).reshape(len(llist), len(zlist), len(zlist))
    power_integrand = power_integrand.diagonal(axis1=1, axis2=2)
    full_integrand = W_chi[np.newaxis, :] * power_integrand
    Cl = simpson(full_integrand, x=chi, axis=1)
    return llist, Cl
