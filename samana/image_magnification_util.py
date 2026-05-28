from lenstronomy.LensModel.Util.decouple_multi_plane_util import setup_grids, coordinates_and_deflections, setup_lens_model
import numpy as np
from scipy.special import hyp2f1 as _hyp2f1, beta as _sp_beta
from lenstronomy.LightModel.light_model import LightModel
from copy import deepcopy
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.Util import util
from lenstronomy.Util.util import make_grid_with_coordtransform
from lenstronomy.Data.coord_transforms import Coordinates
import math as _math

try:
    from numba import njit as _njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Vectorized TNFW ray-shooting helpers
# ---------------------------------------------------------------------------
_TNFW_S = 0.001  # softening scale matching lenstronomy's TNFW._s


def _F_vec(x):
    safe_sub = np.maximum(1.0 - x ** 2, 1e-12)
    safe_sup = np.maximum(x ** 2 - 1.0, 1e-12)
    val_sub = np.arctanh(np.sqrt(safe_sub)) / np.sqrt(safe_sub)
    val_sup = np.arctan(np.sqrt(safe_sup)) / np.sqrt(safe_sup)
    return np.where(x < 1.0, val_sub, np.where(x > 1.0, val_sup, 1.0))


def _L_vec(x, tau):
    x = np.maximum(x, _TNFW_S)
    return np.log(x / (tau + np.sqrt(tau ** 2 + x ** 2)))


def _g_vec(x, tau):
    x = np.maximum(x, _TNFW_S)
    F = _F_vec(x)
    L = _L_vec(x, tau)
    return (
        tau ** 2 * (tau ** 2 + 1.0) ** -2
        * (
            (tau ** 2 + 1.0 + 2.0 * (x ** 2 - 1.0)) * F
            + tau * np.pi
            + (tau ** 2 - 1.0) * np.log(tau)
            + np.sqrt(tau ** 2 + x ** 2) * (-np.pi + L * (tau ** 2 - 1.0) / tau)
        )
    )


def _alpha2rho0(alpha_Rs, Rs):
    return alpha_Rs / (4.0 * Rs ** 2 * (1.0 + np.log(0.5)))


def _batch_tnfw_alpha(theta_x, theta_y, Rs, rho0, r_trunc, cx, cy):
    """Sum TNFW deflections for N halos at M pixel positions in one numpy call.

    theta_x, theta_y : (M,) pixel angular positions [arcsec]
    Rs, rho0, r_trunc, cx, cy : (N,) halo parameters
    Returns alpha_x_sum, alpha_y_sum each of shape (M,)
    """
    x_ = theta_x[None, :] - cx[:, None]   # (N, M)
    y_ = theta_y[None, :] - cy[:, None]   # (N, M)
    R = np.sqrt(x_ ** 2 + y_ ** 2)
    R = np.maximum(R, _TNFW_S * Rs[:, None])
    xi = np.maximum(R / Rs[:, None], _TNFW_S)
    tau = (r_trunc / Rs)[:, None]          # (N, 1)
    gx = _g_vec(xi, tau)                   # (N, M)
    a = 4.0 * rho0[:, None] * Rs[:, None] * gx / xi ** 2
    return np.sum(a * x_, axis=0), np.sum(a * y_, axis=0)


# ---------------------------------------------------------------------------
# Vectorized PSEUDO_DPL (PseudoDoublePowerlaw) helpers
# ---------------------------------------------------------------------------

def _g_pdpl_vec(X, g, n):
    """Vectorized _g for PSEUDO_DPL.

    X: (N, M), g: (N, 1), n: (N, 1)
    Returns (N, M)
    """
    n = np.where(np.abs(n - 3.0) < 1e-6, 3.001, n)
    xi = 1.0 + X ** 2
    a = (n - 3.0) / 2.0
    b = g / 2.0
    c = n / 2.0
    hyp_term = _hyp2f1(a, b, c, 1.0 / xi)
    beta1 = _sp_beta(a, (3.0 - g) / 2.0)
    beta2 = _sp_beta(a, 1.5)
    return 0.5 * (beta1 - beta2 * hyp_term * xi ** ((3.0 - n) / 2.0))


def _alpha2rho0_pdpl(alpha_Rs, Rs, gamma_inner, gamma_outer):
    """Convert alpha_Rs to rho0 for PSEUDO_DPL.  All inputs: (N,)"""
    gx = _g_pdpl_vec(
        np.ones((len(Rs), 1)),
        gamma_inner[:, None],
        gamma_outer[:, None],
    )[:, 0]
    return alpha_Rs / (4.0 * Rs ** 2 * gx)


def _batch_pdpl_alpha(theta_x, theta_y, Rs, rho0, gamma_inner, gamma_outer, cx, cy):
    """Sum PSEUDO_DPL deflections for N halos at M pixel positions.

    theta_x, theta_y : (M,)
    Rs, rho0, gamma_inner, gamma_outer, cx, cy : (N,)
    Returns alpha_x_sum, alpha_y_sum each of shape (M,)
    """
    x_ = theta_x[None, :] - cx[:, None]   # (N, M)
    y_ = theta_y[None, :] - cy[:, None]   # (N, M)
    R = np.maximum(np.sqrt(x_ ** 2 + y_ ** 2), 1e-8)
    xi = np.maximum(R / Rs[:, None], 1e-8)   # X = R/Rs, (N, M)
    gx = _g_pdpl_vec(xi, gamma_inner[:, None], gamma_outer[:, None])   # (N, M)
    a = 4.0 * rho0[:, None] * Rs[:, None] * gx / xi ** 2   # (N, M)
    return np.sum(a * x_, axis=0), np.sum(a * y_, axis=0)


# ---------------------------------------------------------------------------
# Vectorized PointMass helpers
# ---------------------------------------------------------------------------
_PM_S = 1e-25


def _batch_pm_alpha(theta_x, theta_y, theta_E, cx, cy):
    """Sum PointMass deflections for N halos at M pixel positions.

    theta_x, theta_y : (M,)
    theta_E, cx, cy : (N,)
    Returns alpha_x_sum, alpha_y_sum each of shape (M,)
    """
    x_ = theta_x[None, :] - cx[:, None]   # (N, M)
    y_ = theta_y[None, :] - cy[:, None]   # (N, M)
    r2 = np.maximum(x_ ** 2 + y_ ** 2, _PM_S ** 2)
    te2 = theta_E[:, None] ** 2            # (N, 1)
    return np.sum(te2 * x_ / r2, axis=0), np.sum(te2 * y_ / r2, axis=0)


if _NUMBA_AVAILABLE:
    @_njit(cache=True)
    def _F_scalar_nb(x):
        if x < 1.0:
            s = _math.sqrt(max(1.0 - x * x, 1e-12))
            return _math.atanh(s) / s
        elif x > 1.0:
            s = _math.sqrt(max(x * x - 1.0, 1e-12))
            return _math.atan(s) / s
        else:
            return 1.0

    @_njit(cache=True)
    def _g_scalar_nb(xi, tau):
        # scalar TNFW g-function; xi >= _TNFW_S guaranteed by caller
        F = _F_scalar_nb(xi)
        L = _math.log(xi / (tau + _math.sqrt(tau * tau + xi * xi)))
        tau2 = tau * tau
        coeff = tau2 / ((tau2 + 1.0) * (tau2 + 1.0))
        return coeff * (
            (tau2 + 1.0 + 2.0 * (xi * xi - 1.0)) * F
            + tau * _math.pi
            + (tau2 - 1.0) * _math.log(tau)
            + _math.sqrt(tau2 + xi * xi) * (-_math.pi + L * (tau2 - 1.0) / tau)
        )

    @_njit(cache=True)
    def _batch_tnfw_alpha_nb(theta_x, theta_y, Rs, rho0, r_trunc, cx, cy):
        M = theta_x.shape[0]
        N = Rs.shape[0]
        alpha_x = np.zeros(M)
        alpha_y = np.zeros(M)
        for n in range(N):
            rs_n = Rs[n]
            rho0_n = rho0[n]
            tau_n = r_trunc[n] / rs_n
            cx_n = cx[n]
            cy_n = cy[n]
            s_min = _TNFW_S * rs_n
            for m in range(M):
                dx = theta_x[m] - cx_n
                dy = theta_y[m] - cy_n
                R = _math.sqrt(dx * dx + dy * dy)
                if R < s_min:
                    R = s_min
                xi = R / rs_n
                if xi < _TNFW_S:
                    xi = _TNFW_S
                gx = _g_scalar_nb(xi, tau_n)
                a = 4.0 * rho0_n * rs_n * gx / (xi * xi)
                alpha_x[m] += a * dx
                alpha_y[m] += a * dy
        return alpha_x, alpha_y

    @_njit(cache=True)
    def _batch_pm_alpha_nb(theta_x, theta_y, theta_E, cx, cy):
        M = theta_x.shape[0]
        N = theta_E.shape[0]
        alpha_x = np.zeros(M)
        alpha_y = np.zeros(M)
        _PM_S2 = 1e-50  # _PM_S ** 2
        for n in range(N):
            te2 = theta_E[n] * theta_E[n]
            cx_n = cx[n]
            cy_n = cy[n]
            for m in range(M):
                dx = theta_x[m] - cx_n
                dy = theta_y[m] - cy_n
                r2 = dx * dx + dy * dy
                if r2 < _PM_S2:
                    r2 = _PM_S2
                alpha_x[m] += te2 * dx / r2
                alpha_y[m] += te2 * dy / r2
        return alpha_x, alpha_y

    _batch_tnfw_alpha = _batch_tnfw_alpha_nb
    _batch_pm_alpha = _batch_pm_alpha_nb


def _build_tnfw_groups(lens_model_fixed, kwargs_lens_fixed):
    """Preprocess redshift planes for vectorised ray-shooting.

    Returns a list of per-z-plane dicts with:
      tnfw_batch : dict of stacked TNFW params (or None if no TNFW in plane)
      other_k   : list of halo indices for non-TNFW profiles
    """
    mpb = lens_model_fixed.lens_model._multi_plane_base
    sorted_idx = list(mpb._sorted_redshift_index)
    redshift_list = mpb._lens_redshift_list
    T_z_list = mpb._T_z_list
    r2p_list = mpb._reduced2physical_factor
    func_list = mpb.func_list
    T_ij_list = mpb._T_ij_list

    groups = []
    i = 0
    n = len(sorted_idx)
    while i < n:
        k0 = sorted_idx[i]
        z_current = float(redshift_list[k0])
        j = i + 1
        while j < n and abs(float(redshift_list[sorted_idx[j]]) - z_current) < 1e-8:
            j += 1

        tnfw_k, pdpl_k, pm_k, other_k = [], [], [], []
        for m in range(i, j):
            km = sorted_idx[m]
            cname = type(func_list[km]).__name__
            if cname == 'TNFW':
                tnfw_k.append(km)
            elif cname == 'PseudoDoublePowerlaw':
                pdpl_k.append(km)
            elif cname == 'PointMass':
                pm_k.append(km)
            else:
                other_k.append(km)

        tnfw_batch = None
        if tnfw_k:
            Rs = np.array([kwargs_lens_fixed[k]['Rs'] for k in tnfw_k])
            alpha_Rs_arr = np.array([kwargs_lens_fixed[k]['alpha_Rs'] for k in tnfw_k])
            r_trunc = np.array([kwargs_lens_fixed[k]['r_trunc'] for k in tnfw_k])
            cx = np.array([kwargs_lens_fixed[k].get('center_x', 0.0) for k in tnfw_k])
            cy = np.array([kwargs_lens_fixed[k].get('center_y', 0.0) for k in tnfw_k])
            tnfw_batch = {
                'Rs': Rs, 'rho0': _alpha2rho0(alpha_Rs_arr, Rs),
                'r_trunc': r_trunc, 'cx': cx, 'cy': cy,
            }

        pdpl_batch = None
        if pdpl_k:
            Rs = np.array([kwargs_lens_fixed[k]['Rs'] for k in pdpl_k])
            alpha_Rs_arr = np.array([kwargs_lens_fixed[k]['alpha_Rs'] for k in pdpl_k])
            gamma_inner = np.array([kwargs_lens_fixed[k]['gamma_inner'] for k in pdpl_k])
            gamma_outer = np.array([kwargs_lens_fixed[k]['gamma_outer'] for k in pdpl_k])
            cx = np.array([kwargs_lens_fixed[k].get('center_x', 0.0) for k in pdpl_k])
            cy = np.array([kwargs_lens_fixed[k].get('center_y', 0.0) for k in pdpl_k])
            pdpl_batch = {
                'Rs': Rs, 'rho0': _alpha2rho0_pdpl(alpha_Rs_arr, Rs, gamma_inner, gamma_outer),
                'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer,
                'cx': cx, 'cy': cy,
            }

        pm_batch = None
        if pm_k:
            theta_E = np.array([kwargs_lens_fixed[k]['theta_E'] for k in pm_k])
            cx = np.array([kwargs_lens_fixed[k].get('center_x', 0.0) for k in pm_k])
            cy = np.array([kwargs_lens_fixed[k].get('center_y', 0.0) for k in pm_k])
            pm_batch = {'theta_E': theta_E, 'cx': cx, 'cy': cy}

        groups.append({
            'z': z_current,
            'T_z': float(T_z_list[i]),
            'r2p': float(r2p_list[i]),
            'delta_T_stored': float(T_ij_list[i]),
            'tnfw_batch': tnfw_batch,
            'pdpl_batch': pdpl_batch,
            'pm_batch': pm_batch,
            'other_k': other_k,
        })
        i = j
    return groups


def _fast_ray_shoot(mpb, groups, kwargs_lens_fixed, x, y, alpha_x, alpha_y, z_start, z_stop):
    """Vectorised replacement for ray_shooting_partial_comoving."""
    x = np.atleast_1d(np.array(x, dtype=float))
    y = np.atleast_1d(np.array(y, dtype=float))
    alpha_x = np.atleast_1d(np.array(alpha_x, dtype=float))
    alpha_y = np.atleast_1d(np.array(alpha_y, dtype=float))
    z_lens_last = z_start
    first_deflector = True

    for group in groups:
        z_lens = group['z']
        if z_lens <= z_start or z_lens > z_stop:
            continue

        if first_deflector:
            delta_T = (group['delta_T_stored'] if z_start == 0.0
                       else float(mpb._cosmo_bkg.T_xy(z_start, z_lens)))
            first_deflector = False
        else:
            delta_T = group['delta_T_stored']

        x = x + alpha_x * delta_T
        y = y + alpha_y * delta_T
        T_z = group['T_z']
        theta_x = x / T_z
        theta_y = y / T_z
        r2p = group['r2p']

        if group['tnfw_batch'] is not None:
            b = group['tnfw_batch']
            ax, ay = _batch_tnfw_alpha(theta_x, theta_y,
                                       b['Rs'], b['rho0'], b['r_trunc'], b['cx'], b['cy'])
            alpha_x = alpha_x - ax * r2p
            alpha_y = alpha_y - ay * r2p

        if group['pdpl_batch'] is not None:
            b = group['pdpl_batch']
            ax, ay = _batch_pdpl_alpha(theta_x, theta_y,
                                       b['Rs'], b['rho0'], b['gamma_inner'], b['gamma_outer'],
                                       b['cx'], b['cy'])
            alpha_x = alpha_x - ax * r2p
            alpha_y = alpha_y - ay * r2p

        if group['pm_batch'] is not None:
            b = group['pm_batch']
            ax, ay = _batch_pm_alpha(theta_x, theta_y, b['theta_E'], b['cx'], b['cy'])
            alpha_x = alpha_x - ax * r2p
            alpha_y = alpha_y - ay * r2p

        for k in group['other_k']:
            ax, ay = mpb.func_list[k].derivatives(theta_x, theta_y, **kwargs_lens_fixed[k])
            alpha_x = alpha_x - ax * r2p
            alpha_y = alpha_y - ay * r2p

        z_lens_last = z_lens

    delta_T_end = (0.0 if z_lens_last == z_stop
                   else float(mpb._cosmo_bkg.T_xy(z_lens_last, z_stop)))
    x = x + alpha_x * delta_T_end
    y = y + alpha_y * delta_T_end
    return np.asarray(x), np.asarray(y), np.asarray(alpha_x), np.asarray(alpha_y)


def _fast_coords_and_deflections(
    lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
    x_pts, y_pts, z_split, z_source, cosmo_bkg, groups,
):
    """Vectorised replacement for coordinates_and_deflections."""
    mpb = lens_model_fixed.lens_model._multi_plane_base
    Tds = cosmo_bkg.T_xy(z_split, z_source)
    Td = cosmo_bkg.T_xy(0, z_split)
    reduced_to_phys = cosmo_bkg.d_xy(0, z_source) / cosmo_bkg.d_xy(z_split, z_source)

    x_main, y_main, ax_fg, ay_fg = _fast_ray_shoot(
        mpb, groups, kwargs_lens_fixed,
        np.zeros_like(np.atleast_1d(np.asarray(x_pts, dtype=float))),
        np.zeros_like(np.atleast_1d(np.asarray(y_pts, dtype=float))),
        x_pts, y_pts, 0.0, z_split,
    )

    ax_main, ay_main = lens_model_free.alpha(x_main / Td, y_main / Td, kwargs_lens_free)
    ax_main *= reduced_to_phys
    ay_main *= reduced_to_phys

    angle_x = ax_fg - ax_main
    angle_y = ay_fg - ay_main

    x_src, y_src, _, _ = _fast_ray_shoot(
        mpb, groups, kwargs_lens_fixed,
        x_main, y_main, angle_x, angle_y, z_split, z_source,
    )

    ax_bg = (x_src - x_main) / Tds - angle_x
    ay_bg = (y_src - y_main) / Tds - angle_y
    return x_main, y_main, ax_fg, ay_fg, ax_bg, ay_bg


def perturbed_flux_ratios_from_flux_ratios(flux_ratios, flux_ratio_measurement_uncertainties_percentage):
    """

    :param flux_ratios:
    :param flux_ratio_measurement_uncertainties_percentage:
    :return:
    """
    if flux_ratios.ndim == 1:
        flux_ratios_perturbed = [np.random.normal(flux_ratios[i],
                                        flux_ratios[i] *
                                        flux_ratio_measurement_uncertainties_percentage[i]) for i in range(0, 3)]
    else:
        flux_ratios_perturbed = deepcopy(flux_ratios)
        for i in range(0,3):
            flux_ratios_perturbed[:, i] += np.random.normal(0.0,
                                                            flux_ratios_perturbed[:, i] *
                                                            flux_ratio_measurement_uncertainties_percentage[i])
    return np.array(flux_ratios_perturbed)

def perturbed_flux_ratios_from_fluxes(fluxes, flux_measurement_uncertainties_percentage):
    """

    :param fluxes:
    :param flux_measurement_uncertainties_percentage:
    :return:
    """
    fluxes_perturbed = perturbed_fluxes_from_fluxes(fluxes, flux_measurement_uncertainties_percentage)
    fluxes = np.array(fluxes)
    if fluxes.ndim == 1:
        flux_ratios = fluxes_perturbed[1:] / fluxes_perturbed[0]
    else:
        flux_ratios = fluxes_perturbed[:, 1:] / fluxes_perturbed[:,0,np.newaxis]
    return flux_ratios

def perturbed_fluxes_from_fluxes(fluxes, flux_measurement_uncertainties_percentage):
    """

    :param fluxes:
    :param flux_measurement_uncertainties_percentage:
    :return:
    """
    fluxes = np.array(fluxes)
    if fluxes.ndim == 1:
        fluxes_perturbed = []
        for i in range(0, 4):
            df = np.random.normal(0.0, fluxes[i] * flux_measurement_uncertainties_percentage[i])
            fluxes_perturbed.append(fluxes[i] + df)
        fluxes_perturbed = np.array(fluxes_perturbed)
    else:
        fluxes_perturbed = np.empty_like(fluxes)
        for i in range(0, 4):
            df = np.random.normal(0.0, fluxes[:, i] * flux_measurement_uncertainties_percentage[i])
            fluxes_perturbed[:, i] = fluxes[:, i] + df
    return fluxes_perturbed

def magnification_finite_decoupled(source_model, kwargs_source, x_image, y_image,
                                   lens_model_init, kwargs_lens_init, kwargs_lens, index_lens_split,
                                   grid_size_list, grid_resolution,
                                   grid_increment_factor=15.0,
                                   setup_decoupled_multiplane_lens_model_output=None,
                                   magnification_method='CIRCULAR_APERTURE',
                                   rotation_angle_list=None,
                                   hessian_eigenvalue_list=None,
                                   use_vectorized_ray_shooting=True):
    """

    :param source_model:
    :param kwargs_source:
    :param x_image:
    :param y_image:
    :param lens_model_init:
    :param kwargs_lens_init:
    :param kwargs_lens:
    :param index_lens_split:
    :param grid_size_list:
    :param grid_resolution:
    :param grid_increment_factor:
    :param setup_decoupled_multiplane_lens_model_output:
    :param magnification_method:
    :param rotation_angle_list:
    :param hessian_eigenvalue_list:
    :return:
    """
    if setup_decoupled_multiplane_lens_model_output is None:
        lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free, z_source, z_split, cosmo_bkg = \
            setup_lens_model(lens_model_init, kwargs_lens_init, index_lens_split)
    else:
        (lens_model_fixed, lens_model_free, kwargs_lens_fixed,
         kwargs_lens_free, z_source, z_split, cosmo_bkg) = setup_decoupled_multiplane_lens_model_output
    groups = _build_tnfw_groups(lens_model_fixed, kwargs_lens_fixed) if use_vectorized_ray_shooting else None
    magnifications = []
    flux_arrays = []

    for j, (x_img, y_img) in enumerate(zip(x_image, y_image)):

        grid_x_large, grid_y_large, interp_points_large, npix_large = setup_grids(grid_size_list[j],
                                                                                  grid_resolution,
                                                                                  0.0, 0.0)
        grid_x_large = grid_x_large.ravel()
        grid_y_large = grid_y_large.ravel()

        if magnification_method == 'ADAPTIVE':
            mag, flux_array = mag_finite_single_image_adaptive(source_model, kwargs_source, lens_model_fixed,
                                                         lens_model_free, kwargs_lens_fixed, kwargs_lens_free, kwargs_lens,
                                                         z_split, z_source, cosmo_bkg, x_img, y_img, grid_resolution,
                                                         grid_size_list[j],
                                                         z_split, z_source)
            magnifications.append(mag)
            flux_arrays.append(flux_array.T)

        elif magnification_method in ['CIRCULAR_APERTURE', 'ELLIPTICAL_APERTURE']:
            if magnification_method == 'ELLIPTICAL_APERTURE':
                grid_x, grid_y = util.rotate(grid_x_large, grid_y_large,
                                             rotation_angle_list[j])
                grid_r = np.hypot(grid_x, grid_y / hessian_eigenvalue_list[j]).ravel()
            else:
                grid_r = np.hypot(grid_x_large, grid_y_large).ravel()

            r_step = grid_size_list[j] / grid_increment_factor
            mag, flux_array = mag_finite_single_image(source_model, kwargs_source, lens_model_fixed, lens_model_free,
                                                      kwargs_lens_fixed,
                                                      kwargs_lens_free, kwargs_lens, z_split, z_source,
                                                      cosmo_bkg, x_img, y_img, grid_x_large, grid_y_large,
                                                      grid_r, r_step, grid_resolution, grid_size_list[j], z_split, z_source,
                                                      groups=groups)
            magnifications.append(mag)
            flux_arrays.append(flux_array.reshape(npix_large, npix_large))
        else:
            raise Exception('magnification_method must be either CIRCULAR_APERTURE, ELLIPTICAL_APERTURE, or ADAPTIVE. '
                            'You specified magnification_method '+str(magnification_method))

    return np.array(magnifications), flux_arrays

def mag_finite_single_image_adaptive(source_model, kwargs_source, lens_model_fixed, lens_model_free, kwargs_lens_fixed,
                            kwargs_lens_free, kwargs_lens, z_split, z_source,
                            cosmo_bkg, x_image, y_image, grid_resolution, grid_size_max,
                               zlens, zsource, intial_resolution_reduction_factor=4,flux_threshold_factor=200,
                               distance_factor=30):
    """

    :param source_model:
    :param kwargs_source:
    :param lens_model_fixed:
    :param lens_model_free:
    :param kwargs_lens_fixed:
    :param kwargs_lens_free:
    :param kwargs_lens:
    :param z_split:
    :param z_source:
    :param cosmo_bkg:
    :param x_image:
    :param y_image:
    :param grid_resolution:
    :param grid_size_max:
    :param zlens:
    :param zsource:
    :param intial_resolution_reduction_factor:
    :param flux_threshold_factor:
    :param distance_factor:
    :return:
    """
    Td = cosmo_bkg.T_xy(0, zlens)
    Ts = cosmo_bkg.T_xy(0, zsource)
    Tds = cosmo_bkg.T_xy(zlens, zsource)
    reduced_to_phys = cosmo_bkg.d_xy(0, zsource) / cosmo_bkg.d_xy(zlens, zsource)

    # initialize low-res flux array
    deltapix_init = intial_resolution_reduction_factor * grid_resolution
    numPix_init = int(grid_size_max / deltapix_init)
    (grid_x_large_init, grid_y_large_init, ra_at_xy_0, dec_at_xy_0,
     x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix) = (
        make_grid_with_coordtransform(numPix_init, deltapix_init))
    coordinates_lowres = Coordinates(Mpix2coord, ra_at_xy_0, dec_at_xy_0)
    grid_r = np.hypot(grid_x_large_init, grid_y_large_init).ravel()

    # setup ray tracing info
    xD = np.zeros_like(grid_x_large_init)
    yD = np.zeros_like(grid_y_large_init)
    alpha_x_foreground = np.zeros_like(grid_x_large_init)
    alpha_y_foreground = np.zeros_like(grid_y_large_init)
    alpha_x_background = np.zeros_like(grid_x_large_init)
    alpha_y_background = np.zeros_like(grid_y_large_init)
    inds_compute = np.where(grid_r <= grid_size_max/2)
    grid_x_large_init = grid_x_large_init[inds_compute]
    grid_y_large_init = grid_y_large_init[inds_compute]
    x_points_temp = grid_x_large_init + x_image
    y_points_temp = grid_y_large_init + y_image
    _xD, _yD, _alpha_x_foreground, _alpha_y_foreground, _alpha_x_background, _alpha_y_background = \
        coordinates_and_deflections(lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
                                    x_points_temp, y_points_temp, z_split, z_source, cosmo_bkg)
    xD[inds_compute] = _xD
    yD[inds_compute] = _yD
    alpha_x_foreground[inds_compute] = _alpha_x_foreground
    alpha_y_foreground[inds_compute] = _alpha_y_foreground
    alpha_x_background[inds_compute] = _alpha_x_background
    alpha_y_background[inds_compute] = _alpha_y_background
    beta_x, beta_y = calc_source_sb(xD.ravel(),
                                      yD.ravel(),
                                      alpha_x_foreground.ravel(),
                                      alpha_y_foreground.ravel(),
                                      alpha_x_background.ravel(),
                                      alpha_y_background.ravel(),
                                      Td, Tds, Ts, reduced_to_phys,
                                      lens_model_free,
                                      kwargs_lens)
    flux_array = source_model.surface_brightness(beta_x, beta_y, kwargs_source).reshape(numPix_init, numPix_init)

    dist = grid_size_max / distance_factor
    flux_array_threshold = np.max(flux_array) / flux_threshold_factor
    bright_indexes = np.where(flux_array >= flux_array_threshold)
    bright_coords = coordinates_lowres.map_pix2coord(bright_indexes[0], bright_indexes[1])

    # import matplotlib.pyplot as plt
    # plt.imshow(flux_array, origin='upper')
    # plt.show()
    #
    # plt.imshow(flux_array, origin='upper')
    # plt.scatter(bright_indexes[1], bright_indexes[0], color='r',alpha=0.3,marker='+')
    # plt.show()

    # NOW AT HIGH RESOLUTION
    # initialize high-res flux array
    deltapix = grid_resolution
    numPix = int(grid_size_max / deltapix)
    (grid_x_large, grid_y_large, ra_at_xy_0, dec_at_xy_0,
     x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix) = (
        make_grid_with_coordtransform(numPix, deltapix))

    inds_compute_array = np.zeros_like(grid_x_large)
    grid_r = np.hypot(grid_x_large, grid_y_large)
    bright_coords_x, bright_coords_y = bright_coords[1], bright_coords[0]
    for grid_index, (coord_x, coord_y, r_coord) in enumerate(zip(grid_x_large, grid_y_large, grid_r)):
        if r_coord > grid_size_max / 2:
            continue
        else:
            dx, dy = coord_x - bright_coords_x, coord_y - bright_coords_y
            dr = np.sqrt(dx ** 2 + dy ** 2)
            if np.any(dr <= dist):
                inds_compute_array[grid_index] = 1
            # for bright_coord_x, bright_coord_y in zip(bright_coords[1], bright_coords[0]):
            #     dx, dy = coord_x - bright_coord_x, coord_y - bright_coord_y
            #     dr = np.sqrt(dx ** 2 + dy ** 2)
            #     if dr <= dist:
            #         inds_compute_array[grid_index] = 1
            #         pix_x, pix_y = coordinates_highres.map_coord2pix(coord_x, coord_y)
            #         pixel_x_large.append(pix_x)
            #         pixel_y_large.append(pix_y)
            #         break
    inds_compute_array = inds_compute_array.reshape(numPix,numPix)[::-1,::-1]
    #plt.imshow(inds_compute_array, origin='upper'); plt.show()
    inds_compute_highres = np.where(inds_compute_array.ravel()==1)
    x_points_temp = grid_x_large[inds_compute_highres] + x_image
    y_points_temp = grid_y_large[inds_compute_highres] + y_image
    # setup ray tracing info
    xD = np.zeros_like(grid_x_large)
    yD = np.zeros_like(grid_y_large)
    alpha_x_foreground = np.zeros_like(grid_x_large)
    alpha_y_foreground = np.zeros_like(grid_y_large)
    alpha_x_background = np.zeros_like(grid_x_large)
    alpha_y_background = np.zeros_like(grid_y_large)
    _xD, _yD, _alpha_x_foreground, _alpha_y_foreground, _alpha_x_background, _alpha_y_background = \
        coordinates_and_deflections(lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
                                    x_points_temp, y_points_temp, z_split, z_source, cosmo_bkg)
    xD[inds_compute_highres] = _xD
    yD[inds_compute_highres] = _yD
    alpha_x_foreground[inds_compute_highres] = _alpha_x_foreground
    alpha_y_foreground[inds_compute_highres] = _alpha_y_foreground
    alpha_x_background[inds_compute_highres] = _alpha_x_background
    alpha_y_background[inds_compute_highres] = _alpha_y_background
    beta_x_highres, beta_y_highres = calc_source_sb(xD.ravel(),
                                    yD.ravel(),
                                    alpha_x_foreground.ravel(),
                                    alpha_y_foreground.ravel(),
                                    alpha_x_background.ravel(),
                                    alpha_y_background.ravel(),
                                    Td, Tds, Ts, reduced_to_phys,
                                    lens_model_free,
                                    kwargs_lens)
    flux_array_highres = source_model.surface_brightness(beta_x_highres, beta_y_highres, kwargs_source).reshape(numPix, numPix)
    magnification_highres = np.sum(flux_array_highres) * grid_resolution ** 2
    flux_array_highres = flux_array_highres.reshape(numPix, numPix)
    #
    #plt.imshow(flux_array_highres, origin='upper');
    #plt.scatter(pixel_x_large, pixel_y_large, color='r',alpha=0.1,s=5)
    #plt.show()
    #a = input('continue')
    return magnification_highres, flux_array_highres

def calc_source_sb(x, y, alpha_x_foreground, alpha_y_foreground, alpha_x_background, alpha_y_background,
                  Td, Tds, Ts, reduced_to_phys, lens_model_free, kwargs_lens):

    # compute the deflection angles from the main deflector
    deflection_x_main, deflection_y_main = lens_model_free.alpha(
        x / Td, y / Td, kwargs_lens
    )
    deflection_x_main *= reduced_to_phys
    deflection_y_main *= reduced_to_phys

    # add the main deflector to the deflection field
    alpha_x = alpha_x_foreground - deflection_x_main
    alpha_y = alpha_y_foreground - deflection_y_main

    # combine deflections
    alpha_background_x = alpha_x + alpha_x_background
    alpha_background_y = alpha_y + alpha_y_background

    # ray propagation to the source plane with the small angle approximation
    beta_x = x / Ts + alpha_background_x * Tds / Ts
    beta_y = y / Ts + alpha_background_y * Tds / Ts

    return beta_x, beta_y

def _inds_compute_grid_v2(grid_r, r_min, r_max, inds_compute):
    condition1 = grid_r >= r_min
    condition2 = grid_r < r_max
    condition = np.logical_and(condition1, condition2)
    inds_compute_new = np.where(condition)[0]
    inds_outside_r = np.where(grid_r > r_max)[0]
    inds_computed = np.append(inds_compute, inds_compute_new).astype(int)
    return inds_compute_new, inds_outside_r, inds_computed

def decoupled_multiplane_rayshooting(grid_r, r_min, r_max, inds_compute,
                                     grid_x_large, grid_y_large, x_image, y_image,
                                     lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
                                     z_split, z_source, cosmo_bkg, xD, yD, alpha_x_foreground, alpha_y_foreground,
                                     alpha_x_background, alpha_y_background, Td, kwargs_lens, reduced_to_phys,
                                     Ts, Tds, groups=None):
    """
    Ray propagation and image flux calculation with the decoupled multiplane formalism
    :param grid_r:
    :param r_min:
    :param r_max:
    :param inds_compute:
    :param grid_x_large:
    :param grid_y_large:
    :param x_image:
    :param y_image:
    :param lens_model_fixed:
    :param lens_model_free:
    :param kwargs_lens_fixed:
    :param kwargs_lens_free:
    :param z_split:
    :param z_source:
    :param cosmo_bkg:
    :param xD:
    :param yD:
    :param alpha_x_foreground:
    :param alpha_y_foreground:
    :param alpha_x_background:
    :param alpha_y_background:
    :param Td:
    :param kwargs_lens:
    :param reduced_to_phys:
    :param Ts:
    :param Tds:
    :return:
    """
    # select new coordinates to ray-trace through
    inds_compute, inds_outside_r, inds_computed = _inds_compute_grid(grid_r, r_min, r_max, inds_compute)
    x_points_temp = grid_x_large[inds_compute] + x_image
    y_points_temp = grid_y_large[inds_compute] + y_image

    # compute lensing stuff at these coordinates
    if groups is not None:
        _xD, _yD, _alpha_x_foreground, _alpha_y_foreground, _alpha_x_background, _alpha_y_background = \
            _fast_coords_and_deflections(lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
                                        x_points_temp, y_points_temp, z_split, z_source, cosmo_bkg, groups)
    else:
        _xD, _yD, _alpha_x_foreground, _alpha_y_foreground, _alpha_x_background, _alpha_y_background = \
            coordinates_and_deflections(lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
                                        x_points_temp, y_points_temp, z_split, z_source, cosmo_bkg)
    # update the master grids with the new information
    xD[inds_compute] = _xD
    yD[inds_compute] = _yD
    alpha_x_foreground[inds_compute] = _alpha_x_foreground
    alpha_y_foreground[inds_compute] = _alpha_y_foreground
    alpha_x_background[inds_compute] = _alpha_x_background
    alpha_y_background[inds_compute] = _alpha_y_background

    # ray trace to source plane
    x = xD[inds_computed]
    y = yD[inds_computed]
    # compute the deflection angles from the main deflector
    deflection_x_main, deflection_y_main = lens_model_free.alpha(
        x / Td, y / Td, kwargs_lens
    )
    deflection_x_main *= reduced_to_phys
    deflection_y_main *= reduced_to_phys

    # add the main deflector to the deflection field
    alpha_x = alpha_x_foreground[inds_computed] - deflection_x_main
    alpha_y = alpha_y_foreground[inds_computed] - deflection_y_main

    # combine deflections
    alpha_background_x = alpha_x + alpha_x_background[inds_computed]
    alpha_background_y = alpha_y + alpha_y_background[inds_computed]

    # ray propagation to the source plane with the small angle approximation
    beta_x = x / Ts + alpha_background_x * Tds / Ts
    beta_y = y / Ts + alpha_background_y * Tds / Ts
    return beta_x, beta_y, inds_computed, inds_outside_r

def mag_finite_single_image(source_model, kwargs_source, lens_model_fixed, lens_model_free, kwargs_lens_fixed,
                            kwargs_lens_free, kwargs_lens, z_split, z_source,
                            cosmo_bkg, x_image, y_image, grid_x_large, grid_y_large,
                            grid_r, r_step, grid_resolution, grid_size_max, zlens, zsource, groups=None):
    """
    Compute the magnification of a single lensed image with the decoupled multiplane formalism
    :param source_model:
    :param kwargs_source:
    :param lens_model_fixed:
    :param lens_model_free:
    :param kwargs_lens_fixed:
    :param kwargs_lens_free:
    :param kwargs_lens:
    :param z_split:
    :param z_source:
    :param cosmo_bkg:
    :param x_image:
    :param y_image:
    :param grid_x_large:
    :param grid_y_large:
    :param grid_r:
    :param r_step:
    :param grid_resolution:
    :param grid_size_max:
    :param zlens:
    :param zsource:
    :return:
    """
    # initalize flux array
    flux_array = np.zeros(len(grid_x_large))
    # setup ray tracing info
    xD = np.zeros_like(flux_array)
    yD = np.zeros_like(flux_array)
    alpha_x_foreground = np.zeros_like(flux_array)
    alpha_y_foreground = np.zeros_like(flux_array)
    alpha_x_background = np.zeros_like(flux_array)
    alpha_y_background = np.zeros_like(flux_array)
    r_min = 0.0
    r_max = r_min + r_step
    magnification_last = 0.0
    inds_compute = np.array([])
    Td = cosmo_bkg.T_xy(0, zlens)
    Ts = cosmo_bkg.T_xy(0, zsource)
    Tds = cosmo_bkg.T_xy(zlens, zsource)
    reduced_to_phys = cosmo_bkg.d_xy(0, zsource) / cosmo_bkg.d_xy(zlens, zsource)

    while True:

        beta_x, beta_y, inds_computed, inds_outside_r = decoupled_multiplane_rayshooting(grid_r, r_min, r_max, inds_compute,
                                     grid_x_large, grid_y_large, x_image, y_image,
                                     lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
                                     z_split, z_source, cosmo_bkg, xD, yD, alpha_x_foreground, alpha_y_foreground,
                                     alpha_x_background, alpha_y_background, Td, kwargs_lens, reduced_to_phys,
                                     Ts, Tds, groups=groups)

        sb = source_model.surface_brightness(beta_x, beta_y, kwargs_source)
        flux_array[inds_computed] = sb
        flux_array[inds_outside_r] = 0.0

        magnification_temp = np.sum(flux_array) * grid_resolution ** 2
        diff = (
            abs(magnification_temp - magnification_last) / magnification_temp
        )
        r_min += r_step
        r_max += r_step
        if r_max >= grid_size_max:
            break
        elif diff < 0.001 and magnification_temp > 0.0001:  # we want to avoid situations with zero flux
            break
        else:
            magnification_last = magnification_temp
    return magnification_temp, flux_array

def _inds_compute_grid(grid_r, r_min, r_max, inds_compute):
    condition1 = grid_r >= r_min
    condition2 = grid_r < r_max
    condition = np.logical_and(condition1, condition2)
    inds_compute_new = np.where(condition)[0]
    inds_outside_r = np.where(grid_r > r_max)[0]
    inds_computed = np.append(inds_compute, inds_compute_new).astype(int)
    return inds_compute_new, inds_outside_r, inds_computed

def precompute_source_plane_grid(lens_model_init, kwargs_lens_init, kwargs_lens, index_lens_split,
                                 x_image, y_image, grid_size_list, grid_resolution,
                                 setup_decoupled_multiplane_lens_model_output=None,
                                 use_vectorized_ray_shooting=True):
    """
    Pre-compute source-plane coordinates for a fixed (halo realization + macromodel).

    Shoots all rays on a grid of 2x the maximum aperture size around each image at once,
    storing (beta_x, beta_y) per pixel. Subsequent magnification evaluations for different
    source positions/sizes are then cheap vectorized operations with no further ray shooting.

    :param lens_model_init: lens model instance
    :param kwargs_lens_init: lens model kwargs
    :param kwargs_lens: macromodel kwargs (fixed)
    :param index_lens_split: index splitting fixed/free lens components
    :param x_image: image x-coordinates (arcsec)
    :param y_image: image y-coordinates (arcsec)
    :param grid_size_list: list of max aperture radii per image (arcsec)
    :param grid_resolution: pixel size (arcsec)
    :param setup_decoupled_multiplane_lens_model_output: cached lens model setup (optional)
    :param use_vectorized_ray_shooting: use Numba batch ray shooting
    :return: (beta_grids, pixel_offsets)
        beta_grids: list of (beta_x, beta_y) arrays per image, each shape (N_pix,)
        pixel_offsets: list of (dx, dy) image-plane offsets from image center, shape (N_pix,) each.
            Used for aperture masking — pixels near the image center can be selected without
            contamination from neighboring images.
    """
    if setup_decoupled_multiplane_lens_model_output is None:
        lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free, z_source, z_split, cosmo_bkg = \
            setup_lens_model(lens_model_init, kwargs_lens_init, index_lens_split)
    else:
        (lens_model_fixed, lens_model_free, kwargs_lens_fixed,
         kwargs_lens_free, z_source, z_split, cosmo_bkg) = setup_decoupled_multiplane_lens_model_output

    groups = _build_tnfw_groups(lens_model_fixed, kwargs_lens_fixed) if use_vectorized_ray_shooting else None

    Td = cosmo_bkg.T_xy(0, z_split)
    Ts = cosmo_bkg.T_xy(0, z_source)
    Tds = cosmo_bkg.T_xy(z_split, z_source)
    reduced_to_phys = cosmo_bkg.d_xy(0, z_source) / cosmo_bkg.d_xy(z_split, z_source)

    beta_grids = []
    pixel_offsets = []
    for x_img, y_img, grid_size in zip(x_image, y_image, grid_size_list):
        grid_x, grid_y, _, _ = setup_grids(2.0 * grid_size, grid_resolution, 0.0, 0.0)
        grid_x = grid_x.ravel()
        grid_y = grid_y.ravel()

        x_points = grid_x + x_img
        y_points = grid_y + y_img

        if groups is not None:
            xD, yD, ax_fg, ay_fg, ax_bg, ay_bg = _fast_coords_and_deflections(
                lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
                x_points, y_points, z_split, z_source, cosmo_bkg, groups)
        else:
            xD, yD, ax_fg, ay_fg, ax_bg, ay_bg = coordinates_and_deflections(
                lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
                x_points, y_points, z_split, z_source, cosmo_bkg)

        defl_x, defl_y = lens_model_free.alpha(xD / Td, yD / Td, kwargs_lens)
        defl_x *= reduced_to_phys
        defl_y *= reduced_to_phys

        alpha_x = ax_fg - defl_x + ax_bg
        alpha_y = ay_fg - defl_y + ay_bg

        beta_x = xD / Ts + alpha_x * Tds / Ts
        beta_y = yD / Ts + alpha_y * Tds / Ts

        beta_grids.append((beta_x, beta_y))
        pixel_offsets.append((grid_x, grid_y))

    return beta_grids, pixel_offsets


def magnifications_precomputed(beta_grids, grid_resolution, source_x, source_y, source_sigma):
    """
    Compute magnifications from precomputed source-plane coordinates.

    For a fixed (halo realization + macromodel), this replaces repeated ray shooting with
    a single Gaussian evaluation per pixel. Call precompute_source_plane_grid() once, then
    call this function for each (source_x, source_y, source_sigma) combination.

    :param beta_grids: list of (beta_x, beta_y) arrays per image from precompute_source_plane_grid()
    :param grid_resolution: pixel size in arcsec
    :param source_x: source x position (arcsec)
    :param source_y: source y position (arcsec)
    :param source_sigma: Gaussian source size, 1-sigma radius (arcsec)
    :return: array of magnifications, one per image
    """
    magnifications = []
    norm = 2.0 * np.pi * source_sigma ** 2
    for beta_x, beta_y in beta_grids:
        dx = beta_x - source_x
        dy = beta_y - source_y
        flux = np.exp(-(dx * dx + dy * dy) / (2.0 * source_sigma ** 2))
        magnifications.append(np.sum(flux) * grid_resolution ** 2 / norm)
    return np.array(magnifications)


def source_plane_pso(beta_grids, grid_resolution, source_x_init, source_y_init, source_sigma,
                     measured_fluxes, keep_flux_ratio_index,
                     search_window=0.05, n_particles=20, n_iterations=50):
    """
    PSO over source (x, y) position to minimize the flux ratio summary statistic.

    Precompute beta_grids once (via precompute_source_plane_grid), then call this to find
    the source position that best matches the observed flux ratios. Each PSO evaluation is
    a cheap vectorized Gaussian sum with no ray shooting.

    :param beta_grids: list of (beta_x, beta_y) arrays per image, from precompute_source_plane_grid()
    :param grid_resolution: pixel size in arcsec
    :param source_x_init: initial source x position (arcsec) — center of search window
    :param source_y_init: initial source y position (arcsec) — center of search window
    :param source_sigma: Gaussian source 1-sigma radius in arcsec (fixed during PSO)
    :param measured_fluxes: observed flux array (or magnification array), shape (N_images,)
    :param keep_flux_ratio_index: indices into flux_ratios (= measured_fluxes[1:]/measured_fluxes[0]) to use
    :param search_window: half-width of the source position search box in arcsec
    :param n_particles: number of PSO particles
    :param n_iterations: number of PSO iterations
    :return: (magnifications, stat) at best-fit source position
    """
    x_lo = source_x_init - search_window
    x_hi = source_x_init + search_window
    y_lo = source_y_init - search_window
    y_hi = source_y_init + search_window

    fr_data = np.array(measured_fluxes[1:], dtype=float) / measured_fluxes[0]
    keep = list(keep_flux_ratio_index)
    fr_data_keep = np.array([fr_data[i] for i in keep])
    norm = np.max(fr_data_keep)

    def _stat(mags):
        fr_model_keep = np.array([mags[i + 1] / mags[0] for i in keep])
        return np.sqrt(np.sum((fr_data_keep - fr_model_keep) ** 2)) / norm

    rng = np.random.default_rng()
    pos = np.column_stack([
        rng.uniform(x_lo, x_hi, n_particles),
        rng.uniform(y_lo, y_hi, n_particles),
    ])
    vel = np.zeros_like(pos)
    pbest = pos.copy()
    pbest_stat = np.full(n_particles, np.inf)
    gbest = pos[0].copy()
    gbest_stat = np.inf
    gbest_mags = None

    w, c1, c2 = 0.7, 1.5, 1.5

    for _ in range(n_iterations + 1):
        for i in range(n_particles):
            mags = magnifications_precomputed(beta_grids, grid_resolution,
                                             pos[i, 0], pos[i, 1], source_sigma)
            s = _stat(mags)
            if s < pbest_stat[i]:
                pbest_stat[i] = s
                pbest[i] = pos[i].copy()
            if s < gbest_stat:
                gbest_stat = s
                gbest = pos[i].copy()
                gbest_mags = mags.copy()
        r1 = rng.random((n_particles, 2))
        r2 = rng.random((n_particles, 2))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, [x_lo, y_lo], [x_hi, y_hi])

    return gbest_mags, gbest_stat


def setup_gaussian_source(source_fwhm_pc, source_x, source_y, astropy_cosmo, z_source):

    if astropy_cosmo is None:
        from lenstronomy.Cosmo.background import Background
        astropy_cosmo = Background().cosmo
    kpc_per_arcsec = 1/astropy_cosmo.arcsec_per_kpc_proper(z_source).value
    source_sigma = 1e-3 * source_fwhm_pc / 2.354820 / kpc_per_arcsec
    kwargs_source_light = [{'amp': 1.0, 'center_x': source_x, 'center_y': source_y, 'sigma': source_sigma}]
    return LightModel(['GAUSSIAN']), kwargs_source_light
