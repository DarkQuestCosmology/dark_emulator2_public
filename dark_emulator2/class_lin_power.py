# Almost same as Boltzmann linear_solve.py

import numpy as np

from classy import Class
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RectBivariateSpline as rbs
from scipy import constants
from scipy.integrate import quad

from packaging import version
classy_version = '3'

from . import utils as ut

# before 2025/12/15 version
# kmin = -4
# kmax = 2
# nkbin = 2001

# after 2025/12/15 version
kmin = -3
kmax = 2
nkbin = 1001

P_k_max_h = 10**kmax  # for param['P_k_max_h/Mpc']

klist = np.logspace(kmin, kmax, nkbin)
alist = np.append(np.logspace(-2.46, -0.6, 32), np.logspace(-0.6, 0., 81)[1:])
logklist = np.log(klist)
atab = np.logspace(-2.4, 0, 161)


def integrant(lnq, m, T0, a):
    q = np.exp(lnq)
    return q**3 / np.pi**2 * np.sqrt(q**2 + m**2 * a**2) / (np.exp(q / T0) + 1) / a**4


def get_pk(params, trans_totmat0):
    As = np.exp(float(params['ln(10^10As)'])) / 1e10
    ns = float(params['ns'])
    # k0 = float(params['k_pivot']) / (float(params['h0']) / 100)
    k0 = float(params['k_pivot']) / (float(params['h0']))  # Not H0
    return 2 * np.pi**2 / klist**3 * As * (klist / k0)**(ns - 1) * trans_totmat0**2


def get_pk_dimensionless(cosmo, trans_totmat0):
    As = np.exp(float(cosmo.pars['ln10^{10}A_s'])) / 1e10
    ns = float(cosmo.pars['n_s'])
    k0 = float(cosmo.pars['k_pivot']) / (float(cosmo.pars['H0']) / 100)
    return As * (klist / k0)**(ns - 1) * trans_totmat0**2


def get_THwindow(kR):
    return 3 * (np.sin(kR) - kR * np.cos(kR)) / kR**3


def get_sigma8_by_trans(cosmo, trans_totmat0):
    R = 8
    DeltakW = ius(np.log(klist), get_pk_dimensionless(
        cosmo, trans_totmat0) * get_THwindow(klist * R)**2)
    return np.sqrt(quad(lambda x: DeltakW(x),
                        np.log(klist)[0], np.log(klist)[-1], limit=200)[0])


@ut.freezeargs
@ut.functools.lru_cache(maxsize=ut.cache_size)
def prepare_cosmo(cparam, gauge='nbody'):
    cosmo = Class()
    cosmo.set(commonsettings)
    cosmo.set(precisionsettings)
    params = {
        'omega_b': '%g' % cparam['omega_b'], 'omega_cdm': '%g' % cparam['omega_cdm'],
        'm_ncdm': '%g' % (cparam["Mnu"] / 3.0), 'Omega_k': '%g' % cparam['Omega_k'],
        'Omega_Lambda': '0', 'H0': '%g' % (100.0 * cparam['h0']), 'w0_fld': '%g' % cparam['w0'],
        'wa_fld': '%g' % cparam['wa'],
        'ln10^{10}A_s': '%g' % cparam['ln(10^10As)'], 'n_s': '%g' % cparam['ns'],
        'k_pivot': '%g' % cparam['k_pivot'], 'cs2_fld': '%g' % cparam['cs2_fld']
    }
    cosmo.set(params)
    if gauge == 'newtonian':  # default gauge: synchronous
        cosmo.set({'gauge': 'newtonian'})

    cosmo.compute()

    trans = cosmo.get_transfer()
    cbackg = cosmo.get_background()
    cbackg_ascale = 1 / (1 + cbackg['z'])
    rho_ncdm_spl = ius(cbackg_ascale, cbackg['(.)rho_ncdm[0]'])
    w_ncdm_spl = ius(cbackg_ascale, cbackg['(.)p_ncdm[0]'] / cbackg['(.)rho_ncdm[0]'])
    rho_cdm_spl = ius(cbackg_ascale, cbackg['(.)rho_cdm'])
    rho_b_spl = ius(cbackg_ascale, cbackg['(.)rho_b'])
    calH_spl = ius(cbackg_ascale, cbackg['H [1/Mpc]'] * cbackg_ascale)

    tktab_cb = []
    # tktab_nu = []
    tktab_tot = []

    for ascale in alist:
        rho_ncdm = rho_ncdm_spl(ascale)
        w_ncdm = w_ncdm_spl(ascale)
        rho_cdm = rho_cdm_spl(ascale)
        rho_b = rho_b_spl(ascale)
        calH = calH_spl(ascale)
        rho_nonu = rho_cdm + rho_b
        rho_nu = rho_ncdm
        rho_tot = rho_nonu + rho_nu
        zred = 1 / ascale - 1
        trans = cosmo.get_transfer(zred)
        if gauge == 'nbody':
            tc = trans['d_cdm'] + 3 * cosmo.h()**(-2) * calH * trans['t_tot'] / trans['k (h/Mpc)']**2
            tb = trans['d_b'] + 3 * cosmo.h()**(-2) * calH * trans['t_tot'] / trans['k (h/Mpc)']**2
            tncdm = trans['d_ncdm[0]'] + 3 * cosmo.h()**(-2) * calH * (1 + w_ncdm) * trans['t_tot'] / trans['k (h/Mpc)']**2
        else:
            tc = trans['d_cdm']
            tb = trans['d_b']
            tncdm = trans['d_ncdm[0]']
        tktab_cb.append(-(rho_cdm * tc + rho_b * tb) / rho_nonu)
        # tktab_nu.append(-tncdm)
        tktab_tot.append(-(rho_ncdm * tncdm + rho_cdm * tc + rho_b * tb) / rho_tot)
    tktab_cb = np.array(tktab_cb)
    # tktab_nu = np.array(tktab_nu)
    tktab_tot = np.array(tktab_tot)

    cb_spl = rbs(np.log(alist), np.log(trans['k (h/Mpc)']), tktab_cb)
    # nu_spl = rbs(np.log(alist), np.log(trans['k (h/Mpc)']), tktab_nu)
    tot_spl = rbs(np.log(alist), np.log(trans['k (h/Mpc)']), tktab_tot)

    tot_field = tot_spl(0, logklist)[0]  # z=0 , np.log(a=1)
    sig8 = get_sigma8_by_trans(cosmo, tot_field)

    if cparam['sigma8'] is not None:
        As = cparam['As'] * (cparam['sigma8'] / sig8)**2
    else:
        As = cparam['As']

    lnAs = np.log(As * 1e10)
    cosmo.set({'ln10^{10}A_s': '%g' % lnAs})
    cosmo.set({'A_s': '%g' % As})
    cosmo.set({'sigma8': '%g' % sig8})
    cosmo.set({'Mnu': '%g' % cparam["Mnu"]})

    params = get_correct_params_for_emu(cosmo)

    return cb_spl, tot_spl, params


Tcmb = 2.7255  # present CMB temperature in K
Tnu = 0.71611 * Tcmb  # This is for consistency with CLASS default
# present neutrino Temperature in eV
T0 = Tnu * constants.physical_constants['Boltzmann constant in eV/K'][0]

eV_in_kg = constants.physical_constants['electron volt'][0] / \
    constants.physical_constants['speed of light in vacuum'][0]**2
eVinv_in_m = constants.hbar * \
    constants.physical_constants['speed of light in vacuum'][0] / \
    constants.physical_constants['electron volt'][0]
# in convert critical density in the standard h^2 kg m^-3 into [h^2 eV^4] (natural unit)
rho_crit = 1.87847e-26 / eV_in_kg * eVinv_in_m**3


if version.parse(classy_version) >= version.parse('3.0.0'):
    output_setting = 'dTk vTk'
else:
    output_setting = ['dTk', 'vTk']


commonsettings = {
    # This is such that N_eff is adjusted to the standard value of 3.046 following the instruction in "explanatory.ini" attached to CLASS.
    'N_ur': 0.00641,
    'N_ncdm': 1,
    'deg_ncdm': 3,
    'T_cmb': '%g' % Tcmb,
    'tau_reio': '0.079',
    'P_k_max_h/Mpc': P_k_max_h,
    'z_max_pk': 320,
    'output': output_setting,
}


if version.parse(classy_version) >= version.parse('3.0.0'):
    precisionsettings = {  # These are sufficient for this purpose
        'tol_perturbations_integration': 1.e-8,
        'perturbations_sampling_stepsize': 1e-4,
        'tol_ncdm_bg': 1.e-10,
        'background_Nloga': 4620,
    }
else:
    precisionsettings = {  # These are sufficient for this purpose
        'tol_perturb_integration': 1.e-8,
        'perturb_sampling_stepsize': 1e-4,
        'tol_ncdm_bg': 1.e-10,
        # 'background_Nloga': 4620,
    }


def get_correct_params_for_emu(cosmo):
    params = {}
    params["omega_cdm"] = float(cosmo.pars['omega_cdm'])
    params["omega_b"] = cosmo.omega_b()
    params["Omega_m"] = cosmo.Omega_m()
    params["Omega_k"] = cosmo.Omega0_k()
    params["Omega_de"] = 1.0 - cosmo.Omega_m() - cosmo.Omega0_k() - cosmo.Omega_r()

    params["w0"] = float(cosmo.pars['w0_fld'])
    params["wa"] = float(cosmo.pars['wa_fld'])
    params["Mnu"] = float(cosmo.pars['Mnu'])

    params["ns"] = cosmo.n_s()
    params["sigma8"] = float(cosmo.pars['sigma8'])
    params["As"] = float(cosmo.pars['A_s'])
    params["ln(10^10As)"] = float(cosmo.pars['ln10^{10}A_s'])
    params["h0"] = cosmo.h()
    params["k_pivot"] = float(cosmo.pars['k_pivot'])
    params["cs2_fld"] = float(cosmo.pars['cs2_fld'])
    return params


def calc_class_lin_pk(param9d, zred):
    trans_cb, trans_total, params = prepare_cosmo(param9d)

    # "a must be strictly increasing when `grid` is True" error
    a = np.log(1.0 / (zred + 1.0))
    if a.size >= 2 and (a[1] - a[0]) < 0:
        a = a[::-1]
        pk_cb = get_pk(params, trans_cb(a, logklist))[::-1]
        pk_total = get_pk(params, trans_total(a, logklist))[::-1]
    else:
        pk_cb = get_pk(params, trans_cb(a, logklist))
        pk_total = get_pk(params, trans_total(a, logklist))

    return pk_cb, pk_total, params


def get_sigma8(param9d):
    params = prepare_cosmo(param9d)[2]
    return params["sigma8"]


def get_omega_nu(m_ncdm):
    omega_nu = quad(lambda x: integrant(x, m_ncdm, T0, 1.), -
                    14, -3, epsabs=0, epsrel=1e-12, limit=100)[0] / rho_crit
    return omega_nu
