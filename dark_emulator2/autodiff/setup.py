import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import InterpolatedUnivariateSpline as ius

from .. import utils as ut


def spline_matrix(x, x_new):
    # InterpolatedUnivariateSpline(x, y)(x_new) is linear in y, i.e. a fixed matrix for fixed grids
    return np.array([ius(x, e)(x_new) for e in np.eye(len(x))]).T


def gradient_matrix(x):
    return np.array([np.gradient(e, x) for e in np.eye(len(x))]).T


def lin_consts(de, zred, klist):
    lpe = de.lin_pk_emu
    c = {"fid": ut.call_interp2d(lpe.klist, zred[:, np.newaxis], lpe.fid_lin_pk_cb_spl)}
    if klist is not None:
        c["out_T"] = spline_matrix(np.log(lpe.klist), np.log(klist)).T
    return c


def nlin_consts(de, zred, klist):
    pe, lpe = de.pk_emu, de.lin_pk_emu
    args = pe.netargs

    ks, k = pe.support_klist, pe.klist
    c = {"fid_lin": ut.call_interp2d(lpe.klist, zred[:, np.newaxis], lpe.fid_lin_pk_cb_spl),
         "sup_T": spline_matrix(pe.lpe.klist, ks).T,
         "l2n_T": spline_matrix(np.log(lpe.klist), np.log(k)).T,
         "head": np.array([pe.npart / 1024., pe.boxsize / 1000.]),
         "use_log": args.use_log_Pk,
         "param9d": "param9d" in args.input_type,
         "grf": np.ones(pe.support_nklist) if "GRF_delta" in args.input_type else None}

    norm_sup, norm_nl = np.ones((len(zred), len(ks))), np.ones((len(zred), len(k)))
    if args.use_normalized_fiducial_Pk:
        norm_sup = ut.call_interp2d(ks, zred[:, np.newaxis], pe.fid_lin_pk_cb_spl) * np.where(ks <= 0.1, 1.0, (10 * ks)**0.75)
        norm_nl = ut.call_interp2d(k, zred[:, np.newaxis], pe.fid_lin_pk_cb_spl) * np.where(k <= 0.1, 1.0, (10 * k)**0.75)
    if pe.model["type"] == "rand" and pe.model["opt"] == "c_ones":
        norm_nl = norm_nl * ut.call_interp2d(k, zred[:, np.newaxis], pe.calibration_mean)
    c["norm_sup"], c["norm_nl"] = norm_sup, norm_nl

    W = np.ones_like(k)
    if de.lin_pk_switch:
        kidx = np.abs(k - de.lin_pk_switch_k).argmin()
        w = np.linspace(0.0, 1.0, de.nsmooth, endpoint=False)
        W[k <= de.lin_pk_switch_k] = 0.0
        blend = slice(kidx, kidx + de.nsmooth)
        W[blend] = np.where(k[blend] <= de.lin_pk_switch_k, 0.0, w[:len(k[blend])])
    c["W"] = W

    if klist is not None:
        c["out_T"] = spline_matrix(np.log(k), np.log(klist)).T
    return c


def reduce_consts(k, method):
    k = np.asarray(k, dtype=np.float64)
    logk = np.log10(k)
    c = {"logk": logk, "smin": int(np.where(k >= 1e-3)[0][0]), "smax": int(np.where(k < 100)[0][-1]), "type": method["type"]}
    if method["type"] == "pshot":
        c["thr"] = float(np.log10(method["factor"] * method["base"]))
    else:
        c["D_T"] = gradient_matrix(logk).T
        c["thr"] = float(method["factor"])
        c["ign"] = int((k <= 0.8 * method["base"]).sum())
    return c


def limber_consts(zlist, klist, llist, src_dist, dz_max=0.05, n_gauss=8):
    zlist = np.asarray(zlist, dtype=np.float64)
    assert np.diff(zlist)[0] > 0

    # Gauss-Legendre nodes for chi(z_i) = Dh * int_0^{z_i} dz / E(z), as a cumulative weight matrix
    t, wt = np.polynomial.legendre.leggauss(n_gauss)
    edges = np.concatenate([[0.0], zlist])
    nodes, weights, owner = [], [], []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        sub = np.linspace(a, b, max(1, int(np.ceil((b - a) / dz_max))) + 1)
        for a2, b2 in zip(sub[:-1], sub[1:]):
            nodes.append(0.5 * (b2 - a2) * t + 0.5 * (a2 + b2))
            weights.append(0.5 * (b2 - a2) * wt)
            owner.append(np.full(n_gauss, i))
    nodes, weights, owner = (np.concatenate(v) for v in (nodes, weights, owner))
    cum = (owner[np.newaxis, :] <= np.arange(len(zlist))[:, np.newaxis]) * weights[np.newaxis, :]

    c = {"z": zlist, "zq": nodes, "cum_T": cum.T, "k": np.asarray(klist, dtype=np.float64), "l": np.asarray(llist, dtype=np.float64)}
    if src_dist is not None:
        c["nw"] = simpson(np.eye(len(zlist)), x=zlist) * np.asarray(src_dist, dtype=np.float64)
        c["upper"] = np.triu(np.ones((len(zlist), len(zlist))), k=1)
    return c
