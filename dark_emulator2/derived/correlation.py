from . import fftlog


def pk2xi(k, pk, nu=1.01, N_extrap_low=0, N_extrap_high=10000,
          c_window_width=0.25, N_pad=0, kr=1, l=0):
    """
    Compute the two-point correlation function xi(r) from the power spectrum P(k)
    using FFTLog (spherical Hankel transform of order l).

    Parameters
    ----------
    k : array_like
        Wavenumber array (should be logarithmically spaced).
    pk : array_like
        Power spectrum values corresponding to k.
    nu : float, optional
        Biasing parameter for FFTLog (typically close to 1).
    N_extrap_low : int, optional
        Number of extrapolated points at low-k end.
    N_extrap_high : int, optional
        Number of extrapolated points at high-k end.
    c_window_width : float, optional
        Width of cosine window for smoothing Fourier coefficients (fraction).
    N_pad : int, optional
        Number of zero-padding points.
    kr : float, optional
        Characteristic scale kr.
    l : int, optional
        Multipole order (0 for monopole, etc).

    Returns
    -------
    r : ndarray
        Separation distances.
    xi : ndarray
        Two-point correlation function xi(r).
    """

    return fftlog.pk2xi(k, pk, nu, N_extrap_low, N_extrap_high,
                        c_window_width, N_pad, kr, l)


def xi2pk(r, xi, nu=1.01, N_extrap_low=0, N_extrap_high=10000,
          c_window_width=0.25, N_pad=0, kr=1):
    """
    Compute the power spectrum P(k) from two-point correlation function xi(r)
    using inverse FFTLog (inverse spherical Hankel transform).

    Parameters
    ----------
    r : array_like
        Separation distances (should be logarithmically spaced).
    xi : array_like
        Correlation function values at r.
    nu, N_extrap_low, N_extrap_high, c_window_width, N_pad, kr : see pk2xi

    Returns
    -------
    k : ndarray
        Wavenumber array.
    pk : ndarray
        Power spectrum P(k).
    """

    return fftlog.xi2pk(r, xi, nu, N_extrap_low, N_extrap_high,
                        c_window_width, N_pad, kr)


def pk2wp(k, pk, nu=1.01, N_extrap_low=0, N_extrap_high=10000,
          c_window_width=0.25, N_pad=0, kr=1, dlnrp=0.0, D=2):
    """
    Compute the projected correlation function w_p(r_p) from power spectrum P(k),
    via a cylindrical Hankel transform.

    Parameters
    ----------
    k, pk, nu, N_extrap_low, N_extrap_high, c_window_width, N_pad, kr : see pk2xi
    dlnrp : float, optional
        Bin-averaging width in ln(r_p) (for smoothing projected output).
    D : int, optional
        Dimension parameter (typically D=2 for projected correlation function).

    Returns
    -------
    rp : ndarray
        Projected separation distances.
    wp : ndarray
        Projected correlation function w_p(r_p).
    """

    return fftlog.pk2wp(k, pk, nu, N_extrap_low, N_extrap_high,
                        c_window_width, N_pad, kr, dlnrp, D)


def pk2dwp(k, pk, nu=1.01, N_extrap_low=0, N_extrap_high=10000,
           c_window_width=0.25, N_pad=0, kr=1, dlnrp=0.0, D=2):
    """
    Compute the derivative of the projected correlation function dw_p/dr_p
    from power spectrum P(k).

    Parameters
    ----------
    k, pk, nu, N_extrap_low, N_extrap_high, c_window_width, N_pad, kr, dlnrp, D : see pk2wp

    Returns
    -------
    rp : ndarray
        Projected separation distances.
    dwp : ndarray
        Derivative of projected correlation function.
    """

    return fftlog.pk2dwp(k, pk, nu, N_extrap_low, N_extrap_high,
                         c_window_width, N_pad, kr, dlnrp, D)
