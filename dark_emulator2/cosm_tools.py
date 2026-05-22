import numpy as np


def atoz(a):
    """atoz _summary_
    Args:
         a (float or numpy.ndarray):
            scale factor(s) as a float or a numpy array of floats
    Returns:
        z (float or numpy.ndarray):
            redshift(s) as a float or a numpy array of floats
    """
    return (-1.0 + 1.0 / a)


def ztoa(z): return (1.0 / (1.0 + z))


def Pshot(npart, boxsize):
    # npart : number of 1D particle
    # boxsize : box size [Mpc/h]
    return ((boxsize**3) / (npart**3))


def Nyquist_freq(npart, boxsize):
    """Nyquist_freq
    Args:
        npart (int):
            number of 1D particle
        boxsize (float):
            box size [Mpc/h]
    Returns:
        kny (float):
            Nyquist frequency [h/Mpc]
    """
    return (np.pi * (1. * npart / boxsize))
