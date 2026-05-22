# redshift distribution functions n_s(z)

import numpy as np
from scipy.integrate import simpson


def smail_nsz(zlist, z0=0.5, alpha=2.0, beta=1.5, normalize=True):
    """
    Smail-like redshift distribution n_s(z).

    Implements:
        n_s(z) ∝ z^alpha * exp[-(z/z0)^beta]

    Parameters
    ----------
    zlist : ndarray
        Array of redshift values.
    z0 : float, optional
        Characteristic redshift scale.
    alpha : float, optional
        Power-law index.
    beta : float, optional
        Exponential index.
    normalize : bool, optional
        If True, normalize so that ∫ n_s(z) dz = 1.

    Returns
    -------
    nsz : ndarray
        Redshift distribution evaluated at zlist.
    """

    nsz = zlist**alpha * np.exp(- (zlist / z0)**beta)
    if normalize:
        nsz /= simpson(nsz, x=zlist)
    return nsz


def gaussian_nsz(zlist, mu=1.0, sigma=0.1, normalize=True):
    """
    Gaussian redshift distribution n_s(z).

    Implements:
        n_s(z) ∝ exp[-(z - mu)^2 / (2 * sigma^2)]

    Parameters
    ----------
    zlist : ndarray
        Array of redshift values.
    mu : float, optional
        Mean redshift (center of distribution).
    sigma : float, optional
        Standard deviation.
    normalize : bool, optional
        If True, normalize so that ∫ n_s(z) dz = 1.

    Returns
    -------
    nsz : ndarray
        Redshift distribution evaluated at zlist.
    """

    nsz = np.exp(-0.5 * ((zlist - mu) / sigma)**2)
    if normalize:
        nsz /= simpson(nsz, x=zlist)
    return nsz


def tophat_nsz(zlist, zmin=0.9, zmax=1.1, normalize=True):
    """
    Top-hat (box) redshift distribution n_s(z).

    Implements:
        n_s(z) = 1 if zmin <= z <= zmax, else 0

    Parameters
    ----------
    zlist : ndarray
        Array of redshift values.
    zmin : float, optional
        Lower bound of top-hat.
    zmax : float, optional
        Upper bound of top-hat.
    normalize : bool, optional
        If True, normalize so that ∫ n_s(z) dz = 1.

    Returns
    -------
    nsz : ndarray
        Redshift distribution evaluated at zlist.
    """

    nsz = np.where((zlist >= zmin) & (zlist <= zmax), 1.0, 0.0)
    if normalize:
        nsz /= simpson(nsz, x=zlist)
    return nsz


def power_law_nsz(zlist, z0=1.0, alpha=2.0, normalize=True):
    """
    Power-law redshift distribution n_s(z).

    Implements:
        n_s(z) ∝ (z / z0)^alpha for z >= z0, else 0

    Parameters
    ----------
    zlist : ndarray
        Array of redshift values.
    z0 : float, optional
        Threshold redshift; below this n_s(z) = 0.
    alpha : float, optional
        Power-law index.
    normalize : bool, optional
        If True, normalize so that ∫ n_s(z) dz = 1.

    Returns
    -------
    nsz : ndarray
        Redshift distribution evaluated at zlist.
    """

    nsz = (zlist / z0)**alpha
    nsz[zlist < z0] = 0.0
    if normalize:
        nsz /= simpson(nsz, x=zlist)
    return nsz
