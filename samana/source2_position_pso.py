"""
PSO fit of source 2's (the larger source, e.g. the nuclear narrow-line region in a dual-source
quad lens) position offset and size against its own flux ratios.

Source 1 stays the exact lens-equation solution (unchanged elsewhere in the pipeline). Source 2's
position (source_x + offset_x, source_y + offset_y) and size (size_pc_2) are fit here via PSO
against source 2's own measured flux ratios, instead of being drawn blindly from a prior each
forward-model iteration.

See dual_source_pso_notes.md (repo root) for the full design history. Earlier version of this
module precomputed a single near/far split ONCE per image (padded via R_max to cover the whole
PSO search range) and reused it for every candidate, recentering only the flux-collection grid
via a Hessian-predicted shift. That was faster, but an accuracy check against exact ray tracing
(test_source2_pso_accuracy.py) found real failures (up to ~16%) at nonzero offset for some
images: the padding only changes which halos get near/far classified, it does NOT move the
far-field Hessian's fixed expansion point (still centered at the un-shifted image position), so
the far-field's linear approximation breaks down once the grid_shift moves far enough from where
that Hessian was expanded.

Current version: recompute the near/far split fresh for every PSO candidate, centered at that
candidate's own Hessian-predicted image position -- eliminates the fixed-expansion-point problem
by construction, at the cost of redoing the split every evaluation instead of once. Slower per
evaluation (recomputing the split -- and now also the central-ray interpolation -- every time),
but correct across the whole search range, not just near zero offset.
"""
import numpy as np

from samana.image_magnification_near_far import mag_finite_single_image_distortion_adaptive
from samana.image_magnification_util import setup_gaussian_source
from samana.forward_model_util import flux_ratio_summary_statistic, interpolate_ray_paths
from lenstronomy.Util.magnification_finite_util import (
    auto_raytracing_grid_size, auto_raytracing_grid_resolution)


def prior_spec_to_bounds(spec, n_sigma=3):
    """Convert a kwargs_sample_source-style prior spec into (lo, hi) bounds for a PSO search
    window, without sampling from it (see the prior spec format documented in CLAUDE.md /
    forward_model_util.sample_prior).

    :param spec: e.g. ['PSO', lo, hi], ['UNIFORM', lo, hi], ['GAUSSIAN', mean, sigma],
        ['FIXED', value], or ['TRUNC-HALF-GAUSS', mode, sigma, lo, hi]. 'PSO' is the same shape
        as 'UNIFORM' -- it's the tag forward_model_util.sample_prior uses to recognize this key
        should be skipped (not sampled), since it's fit here instead.
    :param n_sigma: for GAUSSIAN specs, bounds are mean +/- n_sigma * sigma
    :return: (lo, hi)
    """
    kind = spec[0]
    if kind == 'FIXED':
        value = spec[1]
        return value, value
    elif kind in ('UNIFORM', 'PSO'):
        _, lo, hi = spec
        return lo, hi
    elif kind == 'GAUSSIAN':
        _, mean, sigma = spec
        return mean - n_sigma * sigma, mean + n_sigma * sigma
    elif kind == 'TRUNC-HALF-GAUSS':
        _, mode, sigma, lo, hi = spec
        return lo, hi
    else:
        raise ValueError('unrecognized prior spec type: ' + str(kind))


def compute_image_hessian_inverse(lens_model_fixed, lens_model_free, kwargs_lens_fixed,
                                  kwargs_lens_free, kwargs_solution, z_split, z_source,
                                  cosmo_bkg, x_image, y_image, eps=1e-4):
    """Per-image inverse Jacobian (A_inv = d(theta)/d(beta)) from a finite-difference Jacobian
    through the decoupled-multiplane machinery (coordinates_and_deflections / calc_source_sb) --
    the SAME helper image_magnification_near_far.exact_point_source_magnification already uses
    for its own finite-difference point-source magnification check.

    Deliberately does NOT use a plain lens_model.hessian(kwargs_solution) call: a LensModel
    configured for that would need kwargs_multiplane_model from the macromodel OPTIMIZER's
    solution, which forward_model_new.py only bakes into a separate `lens_model` object built
    AFTER optimization (forward_model_new.py:641-648) -- that object isn't part of what's passed
    into a magnification_class's __call__. The decoupled fixed/free split
    (setup_decoupled_multiplane_lens_model_output) IS available there, and is exactly what
    coordinates_and_deflections needs, so this reuses that already-correct, already-validated
    path instead.

    Used here to predict, per candidate, where source 2's image actually lands
    (x_image + A_inv @ offset) -- the near/far split gets rebuilt centered there each time (see
    run_source2_pso). sigma_max (largest singular value of A_inv) is also used, in
    PSODoubleGaussianMagnification, to size the PSO's offset/size search bounds dynamically per
    realization from the current local magnification -- see that class's __call__.

    :param lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free: the decoupled
        multiplane split, i.e. setup_decoupled_multiplane_lens_model_output (see
        lenstronomy.LensModel.Util.decouple_multi_plane_util.setup_lens_model)
    :param kwargs_solution: the solved macromodel kwargs -- the actual/proposed macromodel
        (as opposed to kwargs_lens_free, the initial guess used to set up the decoupled
        geometry), passed as calc_source_sb's kwargs_lens argument
    :param x_image, y_image: observed image positions (arcsec)
    :param eps: finite-difference step (arcsec)
    :return: (A_inv_list, sigma_max_list) -- per-image inverse Jacobian and its largest singular
        value (worst-case image-plane displacement per unit source-plane offset, any direction)
    """
    from lenstronomy.LensModel.Util.decouple_multi_plane_util import coordinates_and_deflections
    from samana.image_magnification_util import calc_source_sb

    Td = cosmo_bkg.T_xy(0, z_split)
    Ts = cosmo_bkg.T_xy(0, z_source)
    Tds = cosmo_bkg.T_xy(z_split, z_source)
    reduced_to_phys = cosmo_bkg.d_xy(0, z_source) / cosmo_bkg.d_xy(z_split, z_source)

    A_inv_list = []
    sigma_max_list = []
    for xi, yi in zip(x_image, y_image):
        tx = np.array([xi + eps, xi - eps, xi, xi])
        ty = np.array([yi, yi, yi + eps, yi - eps])
        xD, yD, afx, afy, abx, aby = coordinates_and_deflections(
            lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
            tx, ty, z_split, z_source, cosmo_bkg)
        bx, by = calc_source_sb(xD, yD, afx, afy, abx, aby, Td, Tds, Ts,
                                reduced_to_phys, lens_model_free, kwargs_solution)
        Axx = (bx[0] - bx[1]) / (2 * eps); Ayx = (by[0] - by[1]) / (2 * eps)
        Axy = (bx[2] - bx[3]) / (2 * eps); Ayy = (by[2] - by[3]) / (2 * eps)
        A = np.array([[Axx, Axy], [Ayx, Ayy]])
        A_inv = np.linalg.inv(A)
        A_inv_list.append(A_inv)
        sigma_max_list.append(float(np.linalg.svd(A_inv, compute_uv=False)[0]))
    return A_inv_list, sigma_max_list


def particle_swarm_optimize(objective_fn, lo_bounds, hi_bounds,
                            n_particles=10, n_iterations=30, seed=None,
                            w=0.7, c1=1.5, c2=1.5, verbose=False, verbose_fmt=None):
    """Generic particle swarm optimizer, independent of the lens/magnification machinery.

    :param objective_fn: callable(params) -> (aux, stat), where params is a 1D array matching
        lo_bounds/hi_bounds, stat is the scalar to MINIMIZE, and aux is any extra per-evaluation
        result to keep around for the best-fit particle (e.g. the model magnifications)
    :param lo_bounds, hi_bounds: 1D arrays, the search box
    :param n_particles, n_iterations: swarm size and iteration count
    :param seed: PSO random seed (particle init + velocity update draws), or None
    :param w, c1, c2: inertia / cognitive / social PSO coefficients
    :param verbose: print gbest_stat and gbest params every iteration
    :param verbose_fmt: optional callable(iteration, n_iterations, gbest, gbest_stat) -> str,
        overriding the default verbose print format
    :return: (gbest, gbest_aux, gbest_stat)
    """
    lo_bounds = np.asarray(lo_bounds, dtype=float)
    hi_bounds = np.asarray(hi_bounds, dtype=float)
    n_dim = len(lo_bounds)

    rng = np.random.default_rng(seed)
    pos = rng.uniform(lo_bounds, hi_bounds, size=(n_particles, n_dim))
    vel = np.zeros_like(pos)
    pbest = pos.copy()
    pbest_stat = np.full(n_particles, np.inf)
    gbest = pos[0].copy()
    gbest_stat = np.inf
    gbest_aux = None

    for it in range(n_iterations):
        for p in range(n_particles):
            aux, stat = objective_fn(pos[p])
            if stat < pbest_stat[p]:
                pbest_stat[p] = stat
                pbest[p] = pos[p].copy()
            if stat < gbest_stat:
                gbest_stat = stat
                gbest = pos[p].copy()
                gbest_aux = aux

        r1 = rng.random((n_particles, n_dim))
        r2 = rng.random((n_particles, n_dim))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, lo_bounds, hi_bounds)

        if verbose:
            if verbose_fmt is not None:
                print(verbose_fmt(it, n_iterations, gbest, gbest_stat))
            else:
                print(f'  PSO iter {it + 1}/{n_iterations}  stat={gbest_stat:.4f}  '
                      f'params={np.round(gbest, 5)}')

    return gbest, gbest_aux, gbest_stat


def run_source2_pso(data_class, astropy_cosmo,
                    lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
                    kwargs_solution, z_split, z_source, cosmo_bkg,
                    lens_model_init_batch, kwargs_lens_init_batch,
                    source_x, source_y,
                    rescale_grid_size, rescale_grid_resolution,
                    rotation_angle_list, hessian_eigenvalue_list,
                    offset_x_bounds, offset_y_bounds, size_pc_2_bounds,
                    A_inv_list, halo_masses=None,
                    lens_model_fixed_batched=None, kwargs_lens_fixed_batched=None,
                    n_particles=10, n_iterations=30, seed=None, verbose=False,
                    test_mode=False, R_max_0=0.4,
                    n_coarse=40, rel_tol=5e-4, flux_floor_frac=1e-4,
                    final_n_coarse=None,
                    max_image_plane_shift=0.01):
    """PSO fit of source 2's (offset_x, offset_y, size_pc_2) against its own flux ratios
    (data_class.magnifications[4:8]).

    Per candidate, per image: predicts where that image actually lands
    (shift = A_inv_list[j] @ (offset_x, offset_y)), rebuilds the central-ray interpolation AND
    the near/far split centered at that shifted position (via the un-modified
    mag_finite_single_image_distortion_adaptive wrapper -- same call every production, non-PSO
    evaluation already makes), and uses an un-padded, size-only R_max (no reuse across
    candidates, so no padding is needed -- see module docstring for why the earlier
    padded-and-reused version was inaccurate at nonzero offset).

    This function sets up the lens/magnification-specific objective (compute source 2's 4
    magnifications for a candidate (offset_x, offset_y, size_pc_2), score against fr2 data) and
    hands it to the generic particle_swarm_optimize.

    :param lens_model_init_batch, kwargs_lens_init_batch: the full (non-decoupled), possibly
        batched multiplane model -- used to rebuild interpolate_ray_paths at the shifted image
        position each candidate, same convention image_magnification_util.py's own
        NEAR_FAR_SPLITTING branch uses
    :param source_x, source_y: source 1's exact (lens-equation-solved) position -- source 2's
        candidate position is source_x + offset_x, source_y + offset_y
    :param rescale_grid_size, rescale_grid_resolution: same convention as
        SingleGaussianMagnification -- scalar or per-image list
    :param rotation_angle_list, hessian_eigenvalue_list: static per-image grid-orientation hints
        (e.g. from analysis_util.raytracing_grid_orientation), passed through to the adaptive
        quadtree tiling
    :param offset_x_bounds, offset_y_bounds, size_pc_2_bounds: (lo, hi) PSO search bounds, e.g.
        from prior_spec_to_bounds on the corresponding kwargs_sample_source prior specs
    :param A_inv_list: per-image inverse Jacobian, from compute_image_hessian_inverse
    :param halo_masses: per-halo mass array, same convention as image_magnification_util.py's own
        (un-padded) R_max computation
    :param test_mode: if True, run one extra magnification evaluation at the winning
        (offset_x, offset_y, size_pc_2) after the PSO loop finishes, this time collecting each
        image's flux_array/tiling (skipped during the PSO loop itself to keep every particle
        evaluation cheap). Returned images entry is None when False.
    :param n_coarse, rel_tol, flux_floor_frac: adaptive-quadtree convergence settings, passed
        explicitly to mag_finite_single_image_distortion_adaptive. Defaults match what
        image_magnification_util.py's own dispatcher passes for production single-source
        evaluations (magnification_finite_decoupled's NEAR_FAR_SPLITTING_ADAPTIVE branch) --
        NOT that function's own (looser) defaults, which is what this call used to fall back to
        silently before this was added (n_coarse=20 vs the production n_coarse=40; confirmed via
        check_quadtree_convergence.py that the extra resolution matters -- ~2.4% near-far-vs-exact
        discrepancy at the loose default vs. ~1.8% once both sides are fully converged).
    :param final_n_coarse: if given and different from n_coarse, the search itself still uses
        n_coarse (cheap, e.g. 20 for a ~1.4-1.5x speedup), but ONE extra evaluation at the
        winning (offset_x, offset_y, size_pc_2) is run at final_n_coarse afterward, and its
        magnifications/chi2 replace what the coarse search reported. Cost is a single extra
        evaluation, not per-candidate. Motivated by a 20-random-realization check (see
        conversation history): n_coarse=20 vs n_coarse=40 stayed under ~0.4% at the TRUE point
        for every case, but at the RECOVERED point one case (out of 20) reached 1.11% -- rare,
        but real, so the coarse search's own reported answer isn't reliably trustworthy on its
        own. None (default) preserves the old behavior (single n_coarse throughout). If the
        final-precision evaluation is rejected by either rejection filter (shouldn't happen
        since it's the same point the coarse search already accepted, but checked defensively),
        the coarse result is kept rather than silently discarding a valid answer.
    :param max_image_plane_shift: reject any candidate (offset_x, offset_y, size_pc_2) for which
        any image's PREDICTED image-plane shift (|A_inv_j @ (offset_x, offset_y)|) exceeds this
        (arcsec) -- near-far's accuracy tracks the image-plane shift, not the source-plane
        offset, and a small source-plane offset can produce a much larger image-plane shift for
        a highly magnified image (see the accuracy investigation in dual_source_pso_notes.md).
        Default 0.01 arcsec (10 mas). Rejected candidates get stat=inf without ever running a
        magnification call, same cheap-check-before-expensive-work pattern
        old/optimize_source.py uses for its FWHM prior.
    :return: dict with keys source_x_offset_2, source_y_offset_2, source_size_pc_2,
        source2_pso_chi2, magnifications (best-fit 4-element source-2 magnifications), images
        (per-image [flux_array, tiling] list if test_mode else None)
    """
    x_image = np.asarray(data_class.x_image)
    y_image = np.asarray(data_class.y_image)
    n_images = len(x_image)
    fr2_data = np.asarray(data_class.magnifications)[4:8]
    # source_x/source_y come from lens_model.ray_shooting(data_class.x_image, ...) in
    # forward_model_new.py -- one ray-shot value per image (all ~equal if the lens equation is
    # well-solved). Collapse to a single scalar anchor, same convention
    # SingleGaussianMagnification.__call__ uses (np.mean(source_x), np.mean(source_y)).
    source_x = float(np.mean(source_x))
    source_y = float(np.mean(source_y))
    kpc_per_arcsec = 1.0 / astropy_cosmo.arcsec_per_kpc_proper(z_source).value

    # kwargs_lens_init_batch's macro-deflector entries are the pre-optimization alignment guess
    # (kwargs_lens_align, set in forward_model_new.py), not kwargs_solution -- left as-is here to
    # match main's production convention: forward_model_new.py:587 sets this deliberately, and
    # image_magnification_util.py::magnification_finite_decoupled (main) passes kwargs_lens_batch
    # straight into interpolate_ray_paths unmodified for the single-source NEAR_FAR_SPLITTING
    # path. An earlier version of this function swapped in kwargs_solution here (see
    # near_far_accuracy_investigation_notes.md, repo root) -- that swap was never adopted on main
    # and has been reverted for consistency.

    def compute_mags2(offset_x, offset_y, size_pc_2, return_images=False, n_coarse_override=None):
        # Two independent rejection filters, both checked before any magnification call (same
        # cheap-check-before-expensive-work pattern old/optimize_source.py uses for its FWHM
        # prior) -- a candidate failing EITHER is untrustworthy and gets rejected outright:
        #
        # 1) source-plane offset must not exceed this candidate's OWN proposed size (physical
        #    prior -- old/optimize_source.py's exact check, just re-evaluated per candidate since
        #    size_pc_2 is itself a free PSO parameter here, not fixed).
        offset_mag_arcsec = np.hypot(offset_x, offset_y)
        fwhm_arcsec = 1e-3 * size_pc_2 / kpc_per_arcsec
        if offset_mag_arcsec > fwhm_arcsec:
            return (np.full(n_images, np.nan), None) if return_images else np.full(n_images, np.nan)

        # 2) the PREDICTED image-plane shift (for every image) must stay under
        #    max_image_plane_shift -- near-far's accuracy tracks the image-plane shift, not the
        #    source-plane offset (see dual_source_pso_notes.md's accuracy investigation): a small
        #    source-plane offset can still produce a large, unreliable image-plane shift for a
        #    highly magnified image, via A_inv.
        shifts = [A_inv_list[j] @ np.array([offset_x, offset_y]) for j in range(n_images)]
        if max(np.hypot(*s) for s in shifts) > max_image_plane_shift:
            return (np.full(n_images, np.nan), None) if return_images else np.full(n_images, np.nan)

        src_x = source_x + offset_x
        src_y = source_y + offset_y
        source_model, kwargs_source = setup_gaussian_source(size_pc_2, src_x, src_y,
                                                             astropy_cosmo, z_source)
        grid_size = auto_raytracing_grid_size(size_pc_2)
        if isinstance(rescale_grid_size, (list, np.ndarray)):
            grid_size_list = [r * grid_size for r in rescale_grid_size]
        else:
            grid_size_list = [rescale_grid_size * grid_size] * n_images
        if isinstance(rescale_grid_resolution, (list, np.ndarray)):
            grid_resolution_list = [r * auto_raytracing_grid_resolution(size_pc_2)
                                    for r in rescale_grid_resolution]
        else:
            grid_resolution_list = [rescale_grid_resolution *
                                    auto_raytracing_grid_resolution(size_pc_2)] * n_images

        mags = np.zeros(n_images)
        images = [] if return_images else None
        for j in range(n_images):
            shift = shifts[j]
            x_shifted = x_image[j] + shift[0]
            y_shifted = y_image[j] + shift[1]

            # rebuild the central-ray interpolation AND the near/far split centered at the
            # shifted position -- the whole point of this version: no fixed expansion point to
            # drift away from, since it's recomputed fresh, in the right place, every time.
            ray_interp_x, ray_interp_y = interpolate_ray_paths(
                [x_shifted], [y_shifted], lens_model_init_batch, kwargs_lens_init_batch,
                z_source, terminate_at_source=False)

            rmax_grid_size = grid_size_list[j]
            if halo_masses is None:
                R_max = R_max_0
            else:
                R_max = np.maximum(R_max_0 * (np.array(halo_masses) / 1e8) ** 0.333,
                                   2.0 * rmax_grid_size)
                R_max[np.where(np.log10(halo_masses) > 9)[0]] = 1e9

            rot = rotation_angle_list[j] if rotation_angle_list is not None else None
            hess = hessian_eigenvalue_list[j] if hessian_eigenvalue_list is not None else None
            mag, flux_array, tiling, _ = mag_finite_single_image_distortion_adaptive(
                source_model, kwargs_source, lens_model_fixed, lens_model_free,
                kwargs_lens_fixed, kwargs_solution, z_split, z_source, cosmo_bkg,
                x_shifted, y_shifted, grid_resolution_list[j], grid_size_list[j],
                R_max, ray_interp_x[0], ray_interp_y[0], kwargs_lens_free,
                rotation_angle=rot, hessian_eigenvalue=hess,
                lens_model_fixed_batched=lens_model_fixed_batched,
                kwargs_lens_fixed_batched=kwargs_lens_fixed_batched,
                n_coarse=n_coarse if n_coarse_override is None else n_coarse_override,
                rel_tol=rel_tol, flux_floor_frac=flux_floor_frac)
            mags[j] = mag
            if return_images:
                images.append([flux_array, tiling])  # matches the adaptive-method image
                                                       # convention used elsewhere (see
                                                       # image_magnification_util.py's own
                                                       # NEAR_FAR_SPLITTING_ADAPTIVE branch)
        if return_images:
            return mags, images
        return mags

    def objective_fn(params):
        offset_x, offset_y, size_pc_2 = params
        # test_mode's flux_array/tiling collection only happens once, below, for the winning
        # particle -- not here, so every PSO evaluation stays as cheap as possible.
        mags = compute_mags2(offset_x, offset_y, size_pc_2)
        if np.any(np.isnan(mags)):
            # rejected by the max_image_plane_shift check above -- untrustworthy candidate,
            # never let the PSO treat it as competitive
            return mags, np.inf
        stat, _, _ = flux_ratio_summary_statistic(
            fr2_data, mags, None, data_class.keep_flux_ratio_index,
            data_class.uncertainty_in_fluxes)
        return mags, stat

    def verbose_fmt(it, n_it, gbest, gbest_stat):
        return (f'  source2 PSO iter {it + 1}/{n_it}  stat={gbest_stat:.4f}  '
                f'offset=({gbest[0] * 1e3:.3f}, {gbest[1] * 1e3:.3f}) mas  '
                f'size={gbest[2]:.1f} pc')

    lo_bounds = [offset_x_bounds[0], offset_y_bounds[0], size_pc_2_bounds[0]]
    hi_bounds = [offset_x_bounds[1], offset_y_bounds[1], size_pc_2_bounds[1]]

    gbest, gbest_mags, gbest_stat = particle_swarm_optimize(
        objective_fn, lo_bounds, hi_bounds, n_particles=n_particles,
        n_iterations=n_iterations, seed=seed, verbose=verbose, verbose_fmt=verbose_fmt)

    if gbest_mags is None:
        # every particle, every iteration got rejected by max_image_plane_shift -- the
        # declared (offset_x_bounds, offset_y_bounds) search box is entirely outside the region
        # where near-far is trusted for this image set (common for highly-magnified images;
        # offset_x_bounds/offset_y_bounds may need to shrink, or max_image_plane_shift loosened)
        raise RuntimeError(
            'source2 PSO: every candidate was rejected by max_image_plane_shift='
            f'{max_image_plane_shift} arcsec -- the declared offset bounds '
            f'(x={offset_x_bounds}, y={offset_y_bounds}) may be too wide for at least one '
            'image\'s local magnification (a modest source-plane offset can produce a much '
            'larger predicted image-plane shift near a highly magnified image). Either shrink '
            'the source_x_offset_2/source_y_offset_2 PSO bounds, or loosen max_image_plane_shift.')

    if final_n_coarse is not None and final_n_coarse != n_coarse:
        # one extra full-precision evaluation at the winning point, not per-particle -- buys
        # back the rare (~1/20, up to ~1.2% max mag error) accuracy risk of running the whole
        # search at the cheaper n_coarse (see this function's docstring and the
        # test_speed_settings_20random*.py validation sweep, repo root).
        final_mags = compute_mags2(gbest[0], gbest[1], gbest[2], n_coarse_override=final_n_coarse)
        if not np.any(np.isnan(final_mags)):
            final_stat, _, _ = flux_ratio_summary_statistic(
                fr2_data, final_mags, None, data_class.keep_flux_ratio_index,
                data_class.uncertainty_in_fluxes)
            gbest_mags, gbest_stat = final_mags, final_stat
        # else: the winning point got rejected at final_n_coarse (shouldn't happen -- same
        # rejection filters, same point the coarse search already accepted) -- keep the coarse
        # result rather than silently discarding a valid answer.

    images = None
    if test_mode:
        # one extra evaluation at the winning params, only now collecting flux_array/tiling --
        # cheap relative to the whole PSO run, and keeps every particle evaluation lean.
        eval_n_coarse = final_n_coarse if final_n_coarse is not None else n_coarse
        gbest_mags, images = compute_mags2(gbest[0], gbest[1], gbest[2], return_images=True,
                                           n_coarse_override=eval_n_coarse)

    return {'source_x_offset_2': float(gbest[0]), 'source_y_offset_2': float(gbest[1]),
            'source_size_pc_2': float(gbest[2]), 'source2_pso_chi2': float(gbest_stat),
            'images': images,
            'magnifications': gbest_mags}
