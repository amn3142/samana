"""
Per-plane near/far distortion-field ray tracing.

At each lens plane, halos near the central ray path are kept EXACT; the rest are
collapsed into a single HESSIAN (their convergence+shear at the central ray),
referenced so they produce zero deflection at the center. The grid rays are then
propagated as a *distortion* around the (exactly known) central ray, so the center
ray is undeflected by construction and only the differential distortion -- which is
what magnification depends on -- is accumulated.

Why this fits the cost model: the far halos become 1 HESSIAN per plane, so the
per-grid-point deflection is N_near + 1 profiles instead of N_halos, and it's one
propagation pass over the grid (few ray-shoot calls). Why it should be accurate:
the far field is linearized *per plane* (where it's smooth) about the central ray,
discarding only per-plane flexion -- stronger than a single affine fit made after the
whole far field has accumulated to the image.

Notes on convergence sheets: CONVERGENCE profiles are NOT excluded. A mass sheet is an
exact constant Hessian, so it folds into the far-field HESSIAN term with zero
approximation error; excluding it drops the pyHalo mean-field terms and badly biases
the magnification. The optional kappa-subtraction (subtract_near_kappa /
subtract_far_kappa) is therefore OFF by default -- the macro fit and the pyHalo sheets
already handle the mean field, so subtracting again double-removes it.

VALIDATED: with R_max -> inf (nothing culled) accumulate_distortions reproduces a full
lenstronomy multiplane ray_shooting to ~1e-12 over the grid, and on a real 3124-halo
realization mag_finite_single_image_distortion matches the exact decoupled magnification
to <0.1% at matched integrator.

mag_finite_single_image_distortion is a drop-in for
samana.image_magnification_util.mag_finite_single_image: same growing-annulus driver,
same convergence test and flux bookkeeping, but the per-annulus ray-shoot goes through
the distortion field (central_ray_from_interp once per image,
distortion_multiplane_rayshooting per annulus) instead of decoupled_multiplane_rayshooting.
"""
import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from samana.image_magnification_util import (adaptive_quadtree_magnification,
                                             rasterize_leaves)


_LENS_MODEL_CACHE = {}
_LM_HESSIAN = LensModel(['HESSIAN'])

def _cached_lens_model(names):
    key = tuple(names)
    lm = _LENS_MODEL_CACHE.get(key)
    if lm is None:
        if len(_LENS_MODEL_CACHE) > 5000:   # crude bound for long production runs
            _LENS_MODEL_CACHE.clear()
        lm = LensModel(list(names))
        _LENS_MODEL_CACHE[key] = lm
    return lm

def distortion_field_at_lens_plane(x_co, y_co, x_center_at_plane, y_center_at_plane,
                                   T_z, lens_model_exact, lens_model_far,
                                   kwargs_lens_exact, kwargs_lens_far):
    """Reduced distortion-deflection of the grid rays at one lens plane, referenced to
    the central ray (the deflection at the central ray is zero by construction).

    :param x_co: deviation comoving x of the grid rays from the central ray at this
        plane (the grid ray's absolute angle is x_center_at_plane + x_co / T_z)
    :param y_co: deviation comoving y of the grid rays from the central ray at this plane
    :param x_center_at_plane: angular x position of the central ray at this plane
        (from the precomputed central-ray interpolation)
    :param y_center_at_plane: angular y position of the central ray at this plane
    :param T_z: transverse comoving distance to this plane (Mpc)
    :param lens_model_exact: single-plane LensModel of the near ("exact") halos at z
    :param lens_model_far: single-plane LensModel of the far field -- a single 'HESSIAN'
        centered at (x_center_at_plane, y_center_at_plane) so alpha_far(center) = 0
    :param kwargs_lens_exact: kwargs for lens_model_exact
    :param kwargs_lens_far: kwargs for lens_model_far
    :return: (ax, ay), the reduced distortion-deflection components of the grid rays at
        this plane (zero at the central ray)
    """
    # grid ray's absolute angle = central-ray angle + deviation angle
    theta_x = x_center_at_plane + x_co / T_z
    theta_y = y_center_at_plane + y_co / T_z

    # near halos: exact deflection, minus the value at the central ray (-> distortion)
    ax_exact, ay_exact = lens_model_exact.alpha(theta_x, theta_y, kwargs_lens_exact)
    ax_c, ay_c = lens_model_exact.alpha(x_center_at_plane, y_center_at_plane, kwargs_lens_exact)

    # far halos: linear (Hessian) field, already zero at the center
    ax_far, ay_far = lens_model_far.alpha(theta_x, theta_y, kwargs_lens_far)

    ax = ax_exact + ax_far - ax_c
    ay = ay_exact + ay_far - ay_c

    return ax, ay

def build_plane_index(lens_model_fixed, kwargs_lens_fixed, R_max, exclude_names=()):
    """Parse the fixed model into flat arrays ONCE so that lens_models_at_z does not have
    to walk the full halo list (and rebuild the redshift array) at every lens plane.

    Everything here depends only on (lens_model_fixed, kwargs_lens_fixed, R_max), all of
    which are constant across the planes of an image and across the images of an
    iteration, whereas the central-ray position that lens_models_at_z tests against is
    not. Halos are bucketed by redshift up front, so selecting the halos at a plane costs
    O(n_unique_z) instead of an np.isclose scan over every halo.

    :param lens_model_fixed: the multiplane LensModel of the fixed (halo) deflectors
    :param kwargs_lens_fixed: kwargs for lens_model_fixed
    :param R_max: near/far angular split radius; scalar or per-halo array (see lens_models_at_z)
    :param exclude_names: profile names to skip entirely
    :return: dict consumed by lens_models_at_z
    """
    names = list(lens_model_fixed.lens_model_list)
    zlist = np.asarray(lens_model_fixed.redshift_list, dtype=float)
    n = len(names)

    center_x = np.full(n, np.nan)
    center_y = np.full(n, np.nan)
    for i in range(n):
        kw = kwargs_lens_fixed[i]
        if isinstance(kw, dict):
            center_x[i] = kw.get('center_x', np.nan)
            center_y[i] = kw.get('center_y', np.nan)

    R_max_arr = np.atleast_1d(np.asarray(R_max, dtype=float))
    scalar_rmax = R_max_arr.size == 1
    n_extra = 0 if scalar_rmax else max(n - R_max_arr.size, 0)
    r_split = np.empty(n)
    force_near = np.zeros(n, dtype=bool)
    if scalar_rmax:
        r_split[:] = R_max_arr[0]
    else:
        for i in range(n):
            j = i - n_extra
            if 0 <= j < R_max_arr.size:
                r_split[i] = R_max_arr[j]
            else:
                r_split[i] = np.inf
                force_near[i] = True

    keep = np.ones(n, dtype=bool)
    if exclude_names:
        for i in range(n):
            if names[i] in exclude_names:
                keep[i] = False

    # bucket halo indices by unique redshift; a plane query then only has to np.isclose
    # against the (short) unique-redshift list rather than every halo
    unique_z, inverse = np.unique(zlist, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    bounds = np.concatenate([[0], np.cumsum(np.bincount(inverse, minlength=unique_z.size))])
    buckets = [order[bounds[k]:bounds[k + 1]] for k in range(unique_z.size)]

    return {'names': names, 'center_x': center_x, 'center_y': center_y,
            'r_split': r_split, 'force_near': force_near, 'keep': keep,
            'unique_z': unique_z, 'buckets': buckets, 'n': n}


def _indices_at_z(plane_index, z):
    """Halo indices at redshift z, matching np.where(np.isclose(redshift_list, z))[0]
    exactly (np.isclose on the unique redshifts selects the same halos, since every halo
    in a bucket shares one redshift value).

    :param plane_index: output of build_plane_index
    :param z: lens-plane redshift
    :return: ascending array of indices into lens_model_list
    """
    hits = np.flatnonzero(np.isclose(plane_index['unique_z'], z))
    if hits.size == 0:
        return np.empty(0, dtype=int)
    if hits.size == 1:
        return plane_index['buckets'][hits[0]]
    return np.sort(np.concatenate([plane_index['buckets'][k] for k in hits]))


def _split_near_far(plane_index, at_z, x_center_at_plane, y_center_at_plane):
    """Vectorized near/far partition of the halos at one plane.

    Reproduces the original per-halo predicate exactly, including that only center_x is
    tested for finiteness and that np.hypot (not a squared-distance comparison) sets the
    boundary:

        force_near or (isfinite(center_x) and isfinite(r_split)
                       and hypot(dx, dy) <= r_split)

    :param plane_index: output of build_plane_index
    :param at_z: indices of halos at this plane
    :param x_center_at_plane: angular x of the central ray at this plane
    :param y_center_at_plane: angular y of the central ray at this plane
    :return: (near_idx, far_idx), both ascending
    """
    if at_z.size == 0:
        return at_z, at_z
    at_z = at_z[plane_index['keep'][at_z]]
    if at_z.size == 0:
        return at_z, at_z
    cx = plane_index['center_x'][at_z]
    cy = plane_index['center_y'][at_z]
    r = plane_index['r_split'][at_z]
    with np.errstate(invalid='ignore'):
        within = np.hypot(cx - x_center_at_plane, cy - y_center_at_plane) <= r
    near = plane_index['force_near'][at_z] | (np.isfinite(cx) & np.isfinite(r) & within)
    return at_z[near], at_z[~near]

def precompute_plane_splits(ray_angle_interp_x, ray_angle_interp_y, lens_model_fixed,
                            kwargs_lens_fixed, redshift_planes, cosmo_bkg, R_max, z_source,
                            plane_index=None):
    """Precompute the per-plane near/far split ONCE for an image (it depends only on the
    fixed central ray, not on the annulus grid points, so it is identical for every
    annulus). Reuse the result across annuli to avoid re-splitting the halos and
    re-summing the far-field HESSIAN 10x. This also sidesteps needing a batched HESSIAN:
    the far Hessian is summed here a single time per plane.

    :param ray_angle_interp_x: interp1d(T_z) -> central-ray angular x
    :param ray_angle_interp_y: interp1d(T_z) -> central-ray angular y
    :param lens_model_fixed: multiplane LensModel of the fixed (halo) deflectors
    :param kwargs_lens_fixed: kwargs for lens_model_fixed
    :param redshift_planes: sorted list of lens-plane redshifts to traverse
    :param cosmo_bkg: lenstronomy Background
    :param R_max: near/far split radius (scalar or per-halo array; see lens_models_at_z)
    :param z_source: source redshift
    :return: list of per-plane dicts with keys zi, T_z, x_center, y_center, lm_exact,
        kw_exact, lm_far, kw_far, kappa_near
    """
    splits = []
    d0s = cosmo_bkg.d_xy(0, z_source)
    z_prev = 0.0
    # parse the fixed model once and reuse it at every plane
    if plane_index is None:
        plane_index = build_plane_index(lens_model_fixed, kwargs_lens_fixed, R_max)
    for zi in redshift_planes:
        T_z = cosmo_bkg.T_xy(0, zi)
        x_center = float(ray_angle_interp_x(T_z))
        y_center = float(ray_angle_interp_y(T_z))
        (lm_e, kw_e), (lm_f, kw_f) = lens_models_at_z(
            zi, lens_model_fixed, kwargs_lens_fixed, x_center, y_center, R_max,
            plane_index=plane_index)
        splits.append({'zi': zi,
                       'T_z': T_z,
                       'x_center': x_center,
                       'y_center': y_center,
                       'lm_exact': lm_e,
                       'kw_exact': kw_e,
                       'lm_far': lm_f,
                       'kw_far': kw_f,
                       'delta_T': cosmo_bkg.T_xy(z_prev, zi),
                       'factor': d0s / cosmo_bkg.d_xy(zi, z_source)})
        z_prev = zi
    return splits

def lens_models_at_z(z, lens_model_fixed, kwargs_lens_fixed,
                     x_center_at_plane, y_center_at_plane, R_max,
                     exclude_names=(), verbose=False, plane_index=None):
    """Split the fixed-model halos at redshift z into a near/exact single-plane LensModel
    and a single far-field HESSIAN about the central ray. Distance is measured in the sky
    (angular) plane to the central-ray position at z. A halo is "near" if it has a finite
    center within R_max of the central ray; everything else (including CONVERGENCE sheets,
    which have no center) goes into the far Hessian, where a sheet is represented exactly.

    :param z: the lens-plane redshift to build models for
    :param lens_model_fixed: the multiplane LensModel of the fixed (halo) deflectors
    :param kwargs_lens_fixed: kwargs for lens_model_fixed
    :param x_center_at_plane: angular x position of the central ray at this plane
    :param y_center_at_plane: angular y position of the central ray at this plane
    :param R_max: near/far angular split radius (arcsec). Either a scalar (same radius
        for every halo) or a per-halo array indexed like lens_model_fixed.lens_model_list
        (e.g. mass-scaled, R_max_0 * (M / 1e8)**0.5); halo i is "near" if its center is
        within R_max[i] of the central ray. Non-finite entries (e.g. mass sheets) fall to
        the far field.
    :param exclude_names: profile names to skip entirely. Default () -- nothing is
        excluded. In particular CONVERGENCE sheets are kept (they fold exactly into the
        far Hessian); excluding them drops the pyHalo mean-field and biases the result.
    :param plane_index: optional output of build_plane_index for this (lens_model_fixed,
        kwargs_lens_fixed, R_max). Passing it avoids re-parsing the full halo list at every
        plane; built on the fly when omitted. Must be rebuilt whenever the halo population
        or R_max changes.
    :return: (lens_model_exact, kwargs_lens_exact), (lens_model_far, kwargs_lens_far),
         -- the near/exact model, the single far-field HESSIAN model
    """
    if plane_index is None:
        plane_index = build_plane_index(lens_model_fixed, kwargs_lens_fixed, R_max,
                                       exclude_names=exclude_names)
    names = plane_index['names']
    at_z = _indices_at_z(plane_index, z)
    near_idx, far_idx = _split_near_far(plane_index, at_z,
                                        x_center_at_plane, y_center_at_plane)

    near_names = [names[i] for i in near_idx]
    near_kw = [kwargs_lens_fixed[i] for i in near_idx]

    lens_model_exact = _cached_lens_model(near_names)
    kwargs_lens_exact = near_kw

    # far field: sum the far halos' Hessian at the central ray
    if far_idx.size:
        from samana.forward_model_util import batch_lens_profiles
        far_names_in = [names[i] for i in far_idx]
        far_kw_in = [kwargs_lens_fixed[i] for i in far_idx]
        far_z = [0.0] * len(far_idx)  # single plane; z unused by the hessian
        to_batch = set(far_names_in) - {'CONVERGENCE'}  # don't batch mass sheets
        b_names, _, b_kw = batch_lens_profiles(far_names_in, far_z, far_kw_in,
                                               profiles_to_batch=to_batch, min_group=4)
        lm_far_full = _cached_lens_model(b_names)
        fxx, fxy, fyx, fyy = lm_far_full.hessian(x_center_at_plane, y_center_at_plane, b_kw)
    else:
        fxx = fxy = fyx = fyy = 0.0
    lens_model_far = _LM_HESSIAN
    kwargs_lens_far = [{'f_xx': float(fxx), 'f_xy': float(fxy),
                        'f_yx': float(fyx), 'f_yy': float(fyy),
                        'ra_0': x_center_at_plane, 'dec_0': y_center_at_plane}]
    return (lens_model_exact, kwargs_lens_exact), (lens_model_far, kwargs_lens_far)


def accumulate_distortions(ray_angle_interp_x, ray_angle_interp_y,
                           lens_model_fixed, lens_model_free,
                           kwargs_lens_fixed, kwargs_macro,
                           redshift_planes, x_grid, y_grid,
                           cosmo_bkg, z_source, z_split, R_max,
                           plane_splits=None, kwargs_macro_lin=None, verbose=False):
    """Propagate the grid rays as a distortion around the central ray and return their
    DEVIATION source-plane positions (the caller adds beta_center for the absolute beta).
    Marches plane by plane in comoving coordinates, at each plane splitting the halos
    near/far (lens_models_at_z), applying the reduced distortion-deflection
    (distortion_field_at_lens_plane) with the per-plane reduced->physical factor, and at
    z_split applying the free/macro deflector as a differential relative to the central ray.

    :param ray_angle_interp_x: interp1d(T_z) -> central-ray angular x at that transverse
        comoving distance (from the exact full-model central ray)
    :param ray_angle_interp_y: interp1d(T_z) -> central-ray angular y
    :param lens_model_fixed: multiplane LensModel of the fixed (halo) deflectors
    :param lens_model_free: single-plane LensModel of the free/macro deflector at z_split
    :param kwargs_lens_fixed: kwargs for lens_model_fixed
    :param kwargs_macro: kwargs for lens_model_free -- the CONVERGED macromodel
        (kwargs_solution). This is the deflector at z_split whose shear and convergence
        enter the Jacobian, so it must be the converged model.
    :param redshift_planes: sorted list of lens-plane redshifts to traverse (must include z_split)
    :param x_grid: initial angular x offsets of the grid rays from the central ray
    :param y_grid: initial angular y offsets of the grid rays from the central ray
    :param cosmo_bkg: lenstronomy Background (provides T_xy comoving and d_xy angular distances)
    :param z_source: source redshift
    :param z_split: redshift of the free/macro deflector (the "main" plane)
    :param R_max: near/far angular split radius (scalar; or mass-scale it in lens_models_at_z
    :param kwargs_macro_lin: the macromodel the decoupled multi-plane model was computed with.
    Halos at planes BEHIND z_split are evaluated on the path THIS macro produces
    :return: (dbeta_x, dbeta_y), the deviation source-plane positions of the grid rays
        relative to the central ray
    """

    x_co = np.zeros_like(x_grid, dtype=float)
    y_co = np.zeros_like(y_grid, dtype=float)
    alpha_x = np.array(x_grid, dtype=float)
    alpha_y = np.array(y_grid, dtype=float)

    x_co_lin = y_co_lin = alpha_x_lin = alpha_y_lin = None
    freeze_background = kwargs_macro_lin is not None
    factor_free = cosmo_bkg.d_xy(0, z_source) / cosmo_bkg.d_xy(z_split, z_source)
    if plane_splits is None:
        plane_splits = precompute_plane_splits(ray_angle_interp_x, ray_angle_interp_y,
                                               lens_model_fixed, kwargs_lens_fixed,
                                               redshift_planes, cosmo_bkg, R_max, z_source)
    z_prev = 0.0
    for sp in plane_splits:
        zi = sp['zi']
        T_z = sp['T_z']
        x_center = sp['x_center']
        y_center = sp['y_center']
        # step from the previous plane to this one (comoving)
        x_co = x_co + alpha_x * sp['delta_T']
        y_co = y_co + alpha_y * sp['delta_T']
        if x_co_lin is not None:
            x_co_lin = x_co_lin + alpha_x_lin * sp['delta_T']
            y_co_lin = y_co_lin + alpha_y_lin * sp['delta_T']

        x_sample, y_sample = ((x_co, y_co) if x_co_lin is None
                              else (x_co_lin, y_co_lin))
        dalpha_halo_x, dalpha_halo_y = distortion_field_at_lens_plane(
            x_sample, y_sample, x_center, y_center, T_z,
            sp['lm_exact'], sp['lm_far'], sp['kw_exact'], sp['kw_far'])

        alpha_x = alpha_x - dalpha_halo_x * sp['factor']
        alpha_y = alpha_y - dalpha_halo_y * sp['factor']
        if x_co_lin is not None:
            alpha_x_lin = alpha_x_lin - dalpha_halo_x * sp['factor']
            alpha_y_lin = alpha_y_lin - dalpha_halo_y * sp['factor']

        if np.isclose(zi, z_split):
            theta_ray_x = x_center + x_co / T_z  # grid ray's absolute angle
            theta_ray_y = y_center + y_co / T_z
            alpha_macro_ray_x, alpha_macro_ray_y = lens_model_free.alpha(
                theta_ray_x, theta_ray_y, kwargs_macro)
            alpha_macro_ctr_x, alpha_macro_ctr_y = lens_model_free.alpha(
                x_center, y_center, kwargs_macro)
            if freeze_background:
                alpha_lin_ray_x, alpha_lin_ray_y = lens_model_free.alpha(
                    theta_ray_x, theta_ray_y, kwargs_macro_lin)
                alpha_lin_ctr_x, alpha_lin_ctr_y = lens_model_free.alpha(
                    x_center, y_center, kwargs_macro_lin)
                x_co_lin = np.array(x_co, dtype=float)
                y_co_lin = np.array(y_co, dtype=float)
                # NOTE: computed from alpha_x BEFORE the converged macro is applied below
                alpha_x_lin = alpha_x - (alpha_lin_ray_x - alpha_lin_ctr_x) * factor_free
                alpha_y_lin = alpha_y - (alpha_lin_ray_y - alpha_lin_ctr_y) * factor_free
            alpha_x = alpha_x - (alpha_macro_ray_x - alpha_macro_ctr_x) * factor_free
            alpha_y = alpha_y - (alpha_macro_ray_y - alpha_macro_ctr_y) * factor_free
        z_prev = zi

    T_s = cosmo_bkg.T_xy(0, z_source)
    delta_T = cosmo_bkg.T_xy(z_prev, z_source)
    x_co = x_co + alpha_x * delta_T
    y_co = y_co + alpha_y * delta_T
    dbeta_x = x_co / T_s
    dbeta_y = y_co / T_s
    return dbeta_x, dbeta_y

def _all_halos_at_z(z, lens_model_fixed, kwargs_lens_fixed, exclude_names=()):
    """Build a single-plane LensModel of ALL (non-excluded) fixed halos at redshift z.

    :param z: the lens-plane redshift
    :param lens_model_fixed: the multiplane LensModel of the fixed (halo) deflectors
    :param kwargs_lens_fixed: kwargs for lens_model_fixed
    :param exclude_names: profile names to skip (default (): keep everything)
    :return: (LensModel of the halos at z, list of their kwargs)
    """
    zlist = np.asarray(lens_model_fixed.redshift_list, dtype=float)
    names = lens_model_fixed.lens_model_list
    idx = [i for i in np.where(np.isclose(zlist, z))[0] if names[i] not in exclude_names]
    return LensModel([names[i] for i in idx]), [kwargs_lens_fixed[i] for i in idx]


def central_ray_from_interp(ray_interp_x, ray_interp_y, beta_center, redshift_planes,
                            z_source_convention):
    """Package ray-path interpolators (which the pipeline already has) into the
    central-ray dict consumed by distortion_multiplane_rayshooting / accumulate_distortions.

    The decoupled-multiplane setup (and pyHalo's align_realization) already interpolate
    the image/centroid ray via samana.forward_model_util.interpolate_ray_paths /
    pyHalo.utilities.interpolate_ray_paths, which return interp1d(comoving distance) ->
    angular position. accumulate_distortions queries them at cosmo_bkg.T_xy(0, z), the
    same transverse comoving distance those interpolators use, so they drop in directly.

    :param ray_interp_x: interp1d(T) -> central-ray angular x (e.g. rix[0] from
        interpolate_ray_paths for the image being integrated)
    :param ray_interp_y: interp1d(T) -> central-ray angular y
    :param beta_center: (bx, by) source-plane landing to pin the central ray to. Typically
        the source position (kwargs_source center) so the central ray lands on the source.
    :param redshift_planes: sorted list of lens-plane redshifts to traverse
    :param z_source_convention: redshift the fixed model's reduced deflections are defined against
    :return: dict with keys ray_angle_interp_x, ray_angle_interp_y, beta_center,
        redshift_planes, z_source_convention
    """
    return {'ray_angle_interp_x': ray_interp_x, 'ray_angle_interp_y': ray_interp_y,
            'beta_center': (float(beta_center[0]), float(beta_center[1])),
            'redshift_planes': list(redshift_planes),
            'z_source_convention': z_source_convention}


def setup_central_ray(x_image, y_image, lens_model_full, kwargs_lens_full,
                      redshift_planes, cosmo_bkg, z_source, z_split=None,
                      z_source_convention=None):
    """Convenience wrapper: interpolate the image ray through the FULL multiplane model
    and package it for accumulate_distortions. Calls the pipeline's own
    interpolate_ray_paths (so the path matches how the decoupled setup builds ray paths)
    and takes beta_center from ray_shooting. Prefer central_ray_from_interp when the
    pipeline already has the ray interpolators.

    :param x_image: image angular x position (arcsec)
    :param y_image: image angular y position (arcsec)
    :param lens_model_full: a multiplane LensModel containing ALL deflectors (halos + macro)
    :param kwargs_lens_full: kwargs for lens_model_full
    :param redshift_planes: sorted list of lens-plane redshifts to traverse
    :param cosmo_bkg: lenstronomy Background
    :param z_source: source redshift
    :param z_split: accepted but unused here (the full model already carries every plane;
        z_split matters only inside accumulate_distortions for the free-model plane)
    :param z_source_convention: redshift the reduced deflections are defined against;
        defaults to z_source
    :return: the central-ray dict (see central_ray_from_interp)
    """
    from samana.forward_model_util import interpolate_ray_paths
    if z_source_convention is None:
        z_source_convention = z_source
    rix, riy = interpolate_ray_paths([x_image], [y_image], lens_model_full,
                                     kwargs_lens_full, z_source)
    bx, by = lens_model_full.ray_shooting(x_image, y_image, kwargs_lens_full)
    return central_ray_from_interp(rix[0], riy[0], (bx, by),
                                   redshift_planes, z_source_convention)


def distortion_multiplane_rayshooting(grid_r, r_min, r_max, inds_compute,
                                      grid_x_large, grid_y_large,
                                      lens_model_fixed, lens_model_free,
                                      kwargs_lens_fixed, kwargs_macro,
                                      z_split, z_source, cosmo_bkg, R_max, central_ray,
                                      inds_selector, kwargs_macro_lin=None, verbose=False):
    """Distortion-field analogue of decoupled_multiplane_rayshooting: ray-trace only the
    NEW annulus points [r_min, r_max) and return their source-plane positions. Drops into
    mag_finite_single_image_distortion's growing-region while-loop.

    :param grid_r: radius of every grid point from the image center (arcsec)
    :param r_min: inner radius of the annulus to add this iteration (arcsec)
    :param r_max: outer radius of the annulus to add this iteration (arcsec)
    :param inds_compute: indices already computed on previous iterations (for inds_selector)
    :param grid_x_large: angular x offsets of all grid points from the image center
    :param grid_y_large: angular y offsets of all grid points from the image center
    :param lens_model_fixed: multiplane LensModel of the fixed (halo) deflectors
    :param lens_model_free: single-plane LensModel of the free/macro deflector at z_split
    :param kwargs_lens_fixed: kwargs for lens_model_fixed
    :param kwargs_macro: kwargs for lens_model_free -- the CONVERGED macromodel
    :param z_split: redshift of the free/macro deflector
    :param z_source: source redshift
    :param cosmo_bkg: lenstronomy Background
    :param R_max: near/far angular split radius
    :param central_ray: central-ray dict from central_ray_from_interp / setup_central_ray
        (done once per image)
    :param inds_selector: the annulus point-selection function (e.g. _inds_compute_grid)
    :param kwargs_macro_lin: the aligned macromodel the decoupled Delta-beta was linearized
        at; see accumulate_distortions, if None will use the provided kwargs_solution
    :return: (beta_x_new, beta_y_new, inds_new, inds_outside_r) -- source-plane positions
        of the newly added annulus points, their grid indices, and the indices beyond r_max
    """
    inds_new, inds_outside_r, _ = inds_selector(grid_r, r_min, r_max, inds_compute)
    ox = grid_x_large[inds_new]; oy = grid_y_large[inds_new]     # angular offsets from the image
    dbeta_x, dbeta_y = accumulate_distortions(
        central_ray['ray_angle_interp_x'], central_ray['ray_angle_interp_y'],
        lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_macro,
        central_ray['redshift_planes'], ox, oy, cosmo_bkg, z_source,
        z_split, R_max, plane_splits=central_ray.get('plane_splits'),
        kwargs_macro_lin=kwargs_macro_lin, verbose=verbose)
    bx0, by0 = central_ray['beta_center']
    return bx0 + dbeta_x, by0 + dbeta_y, inds_new, inds_outside_r

def _as_interp(ri):
    """Normalize a ray interpolator argument.

    :param ri: either a single interp1d, or the 1-element list interpolate_ray_paths
        returns (one interp per input coordinate)
    :return: the single interp1d (ri[0] if a list/tuple was passed, else ri)
    """
    if isinstance(ri, (list, tuple)):
        return ri[0]
    return ri

def mag_finite_single_image_distortion_adaptive_from_splits(
        source_model, kwargs_source, lens_model_fixed, lens_model_free, kwargs_lens_fixed,
        kwargs_lens, z_split, z_source, cosmo_bkg, x_image, y_image,
        grid_resolution, grid_size_max, plane_splits, kwargs_lens_free,
        n_coarse=20, rel_tol=1e-3, flux_floor_frac=1e-3,
        rotation_angle=None, hessian_eigenvalue=None,
        lens_model_fixed_batched=None,
        kwargs_lens_fixed_batched=None,
        grid_shift=(0.0, 0.0),
        freeze_background=False
):
    """Adaptive-tiling near/far magnification of a single image, given an ALREADY-BUILT
    near/far split (`plane_splits`, from precompute_plane_splits). This is the flux-integration
    half of mag_finite_single_image_distortion_adaptive, split out so the (expensive,
    R_max-dependent) split can be computed once and reused across many magnification
    evaluations -- e.g. once per image for a whole PSO run, instead of once per particle.

    No R_max param: R_max is only ever used to build plane_splits in the first place: once
    plane_splits is given, accumulate_distortions never falls back to computing one, so R_max
    (and the ray-angle interpolators / redshift_planes used only in that same fallback) are
    dead here and not part of this function's contract.

    :param plane_splits: precomputed near/far split (see precompute_plane_splits), built once
        by the caller -- e.g. by the mag_finite_single_image_distortion_adaptive wrapper below,
        or externally by a caller (like a PSO loop) that wants to reuse the same split across
        many calls with different kwargs_source / grid_shift.
    :param grid_shift: (dx, dy) added to the grid's angular offsets from x_image, y_image before
        ray tracing, so the flux-collection grid can be recentered off the central ray (e.g. onto
        a Hessian-predicted shifted image position for an offset source) without moving the
        central ray/near-far split itself. Default (0.0, 0.0) reproduces the original,
        un-shifted behavior. Applied consistently to both the near-far ray tracing AND the
        exact point-source magnification used for the mu_discrepancy check below, so the two
        are compared at the same (possibly shifted) image-plane location.
    :return: (magnification, flux_array, tiling, mu_discrepancy) -- see
        mag_finite_single_image_distortion_adaptive. tiling['beta_discrepancy'] is a prototype
        diagnostic (offset between beta_center and the true exact beta at this image's position,
        in source-sigma units, nan if kwargs_source has no 'sigma' key) -- see the inline comment
        above its computation.
    """
    lm_exact = lens_model_fixed_batched if lens_model_fixed_batched is not None else lens_model_fixed
    kw_exact = kwargs_lens_fixed_batched if kwargs_lens_fixed_batched is not None else kwargs_lens_fixed

    beta_center = (kwargs_source[0]['center_x'], kwargs_source[0]['center_y'])

    if freeze_background:
        # force ray bundle to match that of initial macromodel guess
        kwargs_macro_lin = kwargs_lens_free
    else:
        # allow ray bundle to match that of kwargs_solution
        kwargs_macro_lin = None

    def ray_shoot(thx, thy):
        ox = np.atleast_1d(thx) - x_image + grid_shift[0]
        oy = np.atleast_1d(thy) - y_image + grid_shift[1]
        # ray_angle_interp_x/y, redshift_planes, R_max are only consulted by
        # accumulate_distortions when plane_splits is None (see docstring above) -- dead here.
        dbeta_x, dbeta_y = accumulate_distortions(
            None, None, lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens,
            None, ox, oy, cosmo_bkg, z_source, z_split, None,
            plane_splits=plane_splits, kwargs_macro_lin=kwargs_macro_lin)
        return beta_center[0] + dbeta_x, beta_center[1] + dbeta_y

    def source_sb(bx, by):
        return source_model.surface_brightness(bx, by, kwargs_source)

    mag, n_calls, n_pts, leaves = adaptive_quadtree_magnification(
        ray_shoot, source_sb, x_image, y_image, grid_size_max, grid_resolution,
        n_coarse=n_coarse, rel_tol=rel_tol, flux_floor_frac=flux_floor_frac,
        rotation_angle=rotation_angle, hessian_eigenvalue=hessian_eigenvalue,
        return_leaves=True)

    npix = max(int(round(grid_size_max / grid_resolution)), 2)
    flux_array = rasterize_leaves(leaves, grid_size_max, npix)
    lcx, lcy, lh, lsb = leaves  # square-quadtree leaves: centers, half-size, SB

    mu_point = exact_point_source_magnification(
        lm_exact, lens_model_free, kw_exact, kwargs_lens_free, kwargs_lens,
        z_split, z_source, cosmo_bkg, x_image + grid_shift[0], y_image + grid_shift[1],
        eps=grid_resolution)

    eps = grid_resolution
    _tx = np.array([x_image + eps, x_image - eps, x_image, x_image])
    _ty = np.array([y_image, y_image, y_image + eps, y_image - eps])
    _bx, _by = ray_shoot(_tx, _ty)
    Axx = (_bx[0] - _bx[1]) / (2 * eps);
    Ayx = (_by[0] - _by[1]) / (2 * eps)
    Axy = (_bx[2] - _bx[3]) / (2 * eps);
    Ayy = (_by[2] - _by[3]) / (2 * eps)
    det = Axx * Ayy - Axy * Ayx
    mu_point_nearfar = 1.0 / abs(det) if det != 0 else np.inf
    mu_discrepancy = (abs(mu_point_nearfar - mu_point) / mu_point
                      if mu_point > 0 and np.isfinite(mu_point_nearfar) else np.inf)

    # beta_discrepancy (PROTOTYPE, not yet wired into any fallback decision): how far near/far's
    # central-ray anchor (beta_center) sits from the TRUE exact beta at this image's own position,
    # in units of the source's own sigma. Complements mu_discrepancy (a JACOBIAN/derivative-only
    # check): the magnification can stay locally smooth and near/far-consistent (mu_discrepancy
    # small) even while the absolute source-plane position near/far assumes has drifted many
    # source-sigma from the truth -- invisible to mu_discrepancy, but catastrophic for a compact
    # source's finite-aperture flux integral (near-zero true flux gets reported as near/far's
    # locally-plausible-looking magnification). Found via seed 50396, WFI2026: mu_discrepancy=0.03
    # (well under the 0.05 tolerance) while the finite-source magnification was wrong by 10 orders
    # of magnitude, traced to a 23.75-sigma beta offset. See near_far_accuracy_investigation_notes.md
    # and debug_seed50396_image2_outlier.py. Stashed in `tiling` (not a new return value) so this
    # stays a non-breaking addition for the many existing 4-tuple-unpacking callers.
    #
    # Deliberately passes kwargs_lens (kwargs_solution) for BOTH of exact_point_source_beta's
    # macro-deflector arguments -- NOT kwargs_lens_free (this function's own parameter, the
    # decoupled-setup's stale initial guess, correctly used elsewhere in this function for the
    # central-ray/ray_interp setup, and also what mu_point above uses for the same "coupling
    # estimate" slot). Using kwargs_lens_free here was tried first and made this check USELESS:
    # it reproduced the same bug documented in exact_point_source_beta's docstring / the
    # near-far-central-ray-bug memory's "Caution" note, collapsing the true 23.75-sigma offset
    # for seed 50396 down to 0.0009 -- i.e. exactly the kind of stale-macromodel mistake this
    # diagnostic exists to catch. mu_point's own use of kwargs_lens_free for this slot may be
    # worth revisiting for the same reason, but that's outside this prototype's scope.
    bx_exact_center, by_exact_center = exact_point_source_beta(
        lm_exact, lens_model_free, kw_exact, kwargs_lens, kwargs_lens,
        z_split, z_source, cosmo_bkg, x_image + grid_shift[0], y_image + grid_shift[1])
    source_sigma = kwargs_source[0].get('sigma', None)
    if source_sigma:
        beta_discrepancy = float(np.hypot(beta_center[0] - bx_exact_center[0],
                                          beta_center[1] - by_exact_center[0]) / source_sigma)
    else:
        beta_discrepancy = np.nan

    tiling = {'shape': 'square', 'cx': lcx, 'cy': lcy, 'half': lh, 'sb': lsb,
              'box_size': grid_size_max, 'npix': npix, 'grid_resolution': grid_resolution,
              'rotation_angle': rotation_angle, 'hessian_eigenvalue': hessian_eigenvalue,
              'n_calls': n_calls, 'n_points': n_pts, 'beta_discrepancy': beta_discrepancy}

    return mag, flux_array, tiling, mu_discrepancy


def mag_finite_single_image_distortion_adaptive(
        source_model, kwargs_source, lens_model_fixed, lens_model_free, kwargs_lens_fixed,
        kwargs_lens, z_split, z_source, cosmo_bkg, x_image, y_image,
        grid_resolution, grid_size_max, R_max, ray_interp_x, ray_interp_y,
        kwargs_lens_free, n_coarse=20, rel_tol=1e-3, flux_floor_frac=1e-3,
        rotation_angle=None, hessian_eigenvalue=None,
        lens_model_fixed_batched=None,
        kwargs_lens_fixed_batched=None,
        freeze_background=False
):
    """Adaptive-tiling version of mag_finite_single_image_distortion. Thin wrapper: builds the
    near/far split (precompute_plane_splits) fresh, then delegates the actual flux integration
    to mag_finite_single_image_distortion_adaptive_from_splits. Preserved for callers that want
    the split computed and consumed in one call, as before; a caller that wants to reuse the
    same split across many magnification evaluations (e.g. a PSO loop) should call
    precompute_plane_splits + mag_finite_single_image_distortion_adaptive_from_splits directly
    instead of going through this wrapper."""

    rix = _as_interp(ray_interp_x); riy = _as_interp(ray_interp_y)
    redshift_planes = sorted(set(np.round(np.asarray(
        lens_model_fixed.redshift_list, dtype=float), 8).tolist()))
    if not any(np.isclose(z, z_split) for z in redshift_planes):
        redshift_planes = sorted(redshift_planes + [float(z_split)])

    plane_splits = precompute_plane_splits(rix, riy, lens_model_fixed, kwargs_lens_fixed,
                                           redshift_planes, cosmo_bkg, R_max, z_source)

    return mag_finite_single_image_distortion_adaptive_from_splits(
        source_model, kwargs_source, lens_model_fixed, lens_model_free, kwargs_lens_fixed,
        kwargs_lens, z_split, z_source, cosmo_bkg, x_image, y_image,
        grid_resolution, grid_size_max, plane_splits, kwargs_lens_free,
        n_coarse=n_coarse, rel_tol=rel_tol, flux_floor_frac=flux_floor_frac,
        rotation_angle=rotation_angle, hessian_eigenvalue=hessian_eigenvalue,
        lens_model_fixed_batched=lens_model_fixed_batched,
        kwargs_lens_fixed_batched=kwargs_lens_fixed_batched,
        freeze_background=freeze_background)


def mag_finite_single_image_distortion(
        source_model, kwargs_source, lens_model_fixed, lens_model_free, kwargs_lens_fixed,
        kwargs_lens, z_split, z_source,
        cosmo_bkg, grid_x_large, grid_y_large,
        grid_r, r_step, grid_resolution, grid_size_max,
        R_max, ray_interp_x, ray_interp_y, x_image, y_image, kwargs_lens_free,
        lens_model_fixed_batched=None,
        kwargs_lens_fixed_batched=None,
        freeze_background=False,
        verbose=False):
    """Finite-source magnification of a single image via the near/far distortion field

    :param source_model: a lenstronomy LightModel for the source
    :param kwargs_source: source light kwargs; kwargs_source[0]['center_x'/'center_y'] IS
        the source position and is used as beta_center (the central ray is pinned here)
    :param lens_model_fixed: multiplane LensModel of the fixed (halo) deflectors
    :param lens_model_free: single-plane LensModel of the free/macro deflector at z_split
    :param kwargs_lens_fixed: kwargs for lens_model_fixed
    :param kwargs_lens: kwargs for the free/macro model (applied at z_split)
    :param z_split: redshift of the free/macro deflector
    :param z_source: source redshift
    :param cosmo_bkg: lenstronomy Background
    :param grid_x_large: angular x offsets of all grid points from the image center
    :param grid_y_large: angular y offsets of all grid points from the image center
    :param grid_r: radius of every grid point from the image center (arcsec)
    :param r_step: annulus growth step (arcsec) for the growing-region loop
    :param grid_resolution: pixel scale (arcsec); magnification = flux_total * grid_resolution**2
    :param grid_size_max: outer bound (arcsec); the loop stops once r_max reaches it
    :param R_max: near/far angular split radius
    :param ray_interp_x: interp1d(T) (or 1-element list) -> central-ray angular x, from
        interpolate_ray_paths on the full initial model
    :param ray_interp_y: interp1d(T) (or 1-element list) -> central-ray angular y
    :param verbose: make print statements
    :return: (magnification, flux_array, log_mag_grad) -- finite-source magnification, the
        per-grid-point surface-brightness array, and log10 of the flux-weighted
        magnification gradient (near-critical trigger; fall back to exact when it
        exceeds ~ -1). -inf if there is no flux.
    """
    rix = _as_interp(ray_interp_x)
    riy = _as_interp(ray_interp_y)
    redshift_planes = sorted(set(np.round(np.asarray(
        lens_model_fixed.redshift_list, dtype=float), 8).tolist()))
    if not any(np.isclose(z, z_split) for z in redshift_planes):
        redshift_planes = sorted(redshift_planes + [float(z_split)])

    lm_exact = lens_model_fixed_batched if lens_model_fixed_batched is not None else lens_model_fixed
    kw_exact = kwargs_lens_fixed_batched if kwargs_lens_fixed_batched is not None else kwargs_lens_fixed

    beta_center = (kwargs_source[0]['center_x'], kwargs_source[0]['center_y'])
    central_ray = central_ray_from_interp(rix, riy, beta_center, redshift_planes, z_source)

    flux_array = np.zeros(len(grid_x_large))
    beta_x_grid = np.full(len(grid_x_large), np.nan)  # source-plane map, for the near-critical check
    beta_y_grid = np.full(len(grid_x_large), np.nan)
    r_min = 0.0
    r_max = r_min + r_step
    magnification_last = 0.0
    inds_compute = np.array([])
    flux_total = 0.0
    central_ray['plane_splits'] = precompute_plane_splits(
        rix, riy, lens_model_fixed, kwargs_lens_fixed, redshift_planes, cosmo_bkg, R_max, z_source)

    if freeze_background:
        # force ray bundle to match that of initial macromodel guess
        kwargs_macro_lin = kwargs_lens_free
    else:
        # allow ray bundle to match that of kwargs_solution
        kwargs_macro_lin = None
    while True:
        beta_x_new, beta_y_new, inds_new, _ = distortion_multiplane_rayshooting(
            grid_r, r_min, r_max, inds_compute,
            grid_x_large, grid_y_large,
            lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens,
            z_split, z_source, cosmo_bkg, R_max, central_ray, _inds_compute_grid,
            kwargs_macro_lin=kwargs_macro_lin, verbose=verbose)
        sb_new = source_model.surface_brightness(beta_x_new, beta_y_new, kwargs_source)
        flux_array[inds_new] = sb_new
        beta_x_grid[inds_new] = beta_x_new
        beta_y_grid[inds_new] = beta_y_new
        flux_total += np.sum(sb_new)
        magnification_temp = flux_total * grid_resolution ** 2
        diff = abs(magnification_temp - magnification_last) / magnification_temp if magnification_temp > 0 else 1.0
        r_min += r_step
        r_max += r_step

        if r_max >= grid_size_max:
            break
        elif diff < 0.001 and magnification_temp > 0.0001:
            break
        else:
            magnification_last = magnification_temp

    mu_point = exact_point_source_magnification(
        lm_exact, lens_model_free, kw_exact, kwargs_lens_free, kwargs_lens,
        z_split, z_source, cosmo_bkg, x_image, y_image, eps=grid_resolution)
    mu_discrepancy = abs(magnification_temp - mu_point) / mu_point if mu_point > 0 else np.inf

    # beta_discrepancy -- see the identical, more-detailed comment in
    # mag_finite_single_image_distortion_adaptive_from_splits, including why this deliberately
    # passes kwargs_lens (not kwargs_lens_free) for BOTH macro-deflector arguments.
    bx_exact_center, by_exact_center = exact_point_source_beta(
        lm_exact, lens_model_free, kw_exact, kwargs_lens, kwargs_lens,
        z_split, z_source, cosmo_bkg, x_image, y_image)
    source_sigma = kwargs_source[0].get('sigma', None)
    if source_sigma:
        beta_discrepancy = float(np.hypot(beta_center[0] - bx_exact_center[0],
                                          beta_center[1] - by_exact_center[0]) / source_sigma)
    else:
        beta_discrepancy = np.nan

    return magnification_temp, flux_array, mu_discrepancy, beta_discrepancy

def exact_point_source_beta(lens_model_fixed, lens_model_free, kwargs_lens_fixed,
                            kwargs_lens_free, kwargs_lens, z_split, z_source,
                            cosmo_bkg, x, y):
    """Exact decoupled-multiplane source-plane position(s) beta(x, y), via the same
    production 'exact' path (coordinates_and_deflections + calc_source_sb) used by
    exact_point_source_magnification below. x, y may be scalars or arrays; kwargs_lens
    must be the actual solved macromodel (not a pre-optimization guess -- see
    near_far_accuracy_investigation_notes.md, Part 1).

    :return: (beta_x, beta_y) arrays, same shape as x, y
    """
    from lenstronomy.LensModel.Util.decouple_multi_plane_util import coordinates_and_deflections
    from samana.image_magnification_util import calc_source_sb
    Td = cosmo_bkg.T_xy(0, z_split); Ts = cosmo_bkg.T_xy(0, z_source)
    Tds = cosmo_bkg.T_xy(z_split, z_source)
    reduced_to_phys = cosmo_bkg.d_xy(0, z_source) / cosmo_bkg.d_xy(z_split, z_source)
    tx = np.atleast_1d(x); ty = np.atleast_1d(y)
    xD, yD, afx, afy, abx, aby = coordinates_and_deflections(
        lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
        tx, ty, z_split, z_source, cosmo_bkg)
    return calc_source_sb(xD, yD, afx, afy, abx, aby, Td, Tds, Ts,
                          reduced_to_phys, lens_model_free, kwargs_lens)


def source_position_convergence_check(lens_model_fixed, lens_model_free, kwargs_lens_fixed,
                                      kwargs_lens_free, kwargs_lens, z_split, z_source,
                                      cosmo_bkg, x_image, y_image, source_sigma,
                                      threshold=0.05, verbose=False):
    """Standalone diagnostic (NOT called from the production near/far path) for whether
    anchoring the central ray to kwargs_source's shared average center -- what
    mag_finite_single_image_distortion(_adaptive_from_splits) actually do -- is a safe
    approximation for a source of size source_sigma, for a given solved macromodel.

    Ray-traces each image's OWN exact source-plane position via exact_point_source_beta (the
    same production 'exact' path used elsewhere), then looks at how far apart those per-image
    solutions land from EACH OTHER -- the spread -- not any one image's offset from their
    average. For a well-converged macromodel the images agree to ~1e-7 arcsec; when the
    optimizer's internal convergence check reports "converged" without matching true
    (non-linearized) physics, they can disagree by ~1e-4 arcsec or more. A synthetic-offset
    sweep found this starts to matter once the spread is a non-negligible fraction of the
    source's own sigma: negligible below ~0.01 sigma (quadtree noise floor), >1% flux error by
    ~0.1 sigma, ~8% by 0.5 sigma. See near_far_accuracy_investigation_notes.md (repo root,
    Parts 2-3) for the full investigation, including a broader 80-seed sweep with real (not
    injected) macromodel scatter.

    :param x_image, y_image: image positions (arcsec), one per image
    :param source_sigma: the source's Gaussian sigma (arcsec), e.g. kwargs_source[0]['sigma']
    :param threshold: spread/sigma ratio above which convergence is flagged as a real risk to
        near/far's average-anchoring approximation (~0.05-0.1 per the investigation)
    :return: dict with 'beta_x'/'beta_y' (each image's own exact source position), 'spread'
        (max pairwise distance among them, arcsec), 'ratio' (spread / source_sigma), and
        'flagged' (bool)
    """
    bx, by = exact_point_source_beta(lens_model_fixed, lens_model_free, kwargs_lens_fixed,
                                     kwargs_lens_free, kwargs_lens, z_split, z_source,
                                     cosmo_bkg, np.asarray(x_image), np.asarray(y_image))
    spread = 0.0
    for i in range(len(bx)):
        for j in range(i + 1, len(bx)):
            spread = max(spread, float(np.hypot(bx[i] - bx[j], by[i] - by[j])))
    ratio = spread / source_sigma
    flagged = ratio > threshold
    if verbose:
        print('source position spread across images: {:.3e} arcsec ({:.4f} x source sigma) '
             '-- {} (threshold {})'.format(
                 spread, ratio, 'FLAGGED' if flagged else 'ok', threshold))
    return {'beta_x': bx, 'beta_y': by, 'spread': spread, 'ratio': ratio,
           'flagged': flagged, 'threshold': threshold}


def exact_point_source_magnification(lens_model_fixed, lens_model_free, kwargs_lens_fixed,
                                     kwargs_lens_free, kwargs_lens, z_split, z_source,
                                     cosmo_bkg, x_image, y_image, eps):
    """Exact point-source magnification 1/|det(dbeta/dtheta)| at (x_image, y_image), from a
    5-point finite-difference Hessian through the decoupled multiplane model (~5 ray-shoots).
    For a small source away from a caustic the finite-source magnification equals this, so a
    large near/far-vs-this discrepancy flags a near/far failure (over- OR under-production)
    that the geometry-based fold metric misses."""
    tx = np.array([x_image + eps, x_image - eps, x_image, x_image])
    ty = np.array([y_image, y_image, y_image + eps, y_image - eps])
    bx, by = exact_point_source_beta(lens_model_fixed, lens_model_free, kwargs_lens_fixed,
                                     kwargs_lens_free, kwargs_lens, z_split, z_source,
                                     cosmo_bkg, tx, ty)
    Axx = (bx[0] - bx[1]) / (2 * eps); Ayx = (by[0] - by[1]) / (2 * eps)
    Axy = (bx[2] - bx[3]) / (2 * eps); Ayy = (by[2] - by[3]) / (2 * eps)
    det = Axx * Ayy - Axy * Ayx
    return 1.0 / abs(det) if det != 0 else np.inf

def _inds_compute_grid(grid_r, r_min, r_max, inds_compute):
    """Select the grid points in the annulus [r_min, r_max) to add this iteration.

    :param grid_r: radius of every grid point from the image center (arcsec)
    :param r_min: inner annulus radius (arcsec)
    :param r_max: outer annulus radius (arcsec)
    :param inds_compute: indices computed on previous iterations
    :return: (inds_compute_new, inds_outside_r, inds_computed) -- indices newly selected
        in the annulus, indices beyond r_max, and the accumulated set of computed indices
    """
    condition = np.logical_and(grid_r >= r_min, grid_r < r_max)
    inds_compute_new = np.where(condition)[0]
    inds_outside_r = np.where(grid_r > r_max)[0]
    inds_computed = np.append(inds_compute, inds_compute_new).astype(int)
    return inds_compute_new, inds_outside_r, inds_computed
