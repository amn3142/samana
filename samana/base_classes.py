from copy import deepcopy

from samana.image_magnification_util import setup_gaussian_source
from samana.forward_model_util import flux_ratio_summary_statistic
from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_size, auto_raytracing_grid_resolution
from samana.source2_position_pso import (prior_spec_to_bounds, compute_image_hessian_inverse,
                                         run_source2_pso)

import numpy as np


class DarkMatterBaseClass(object):

    def __init__(self, astropy_cosmo):
        """
        astropy_cosmo: an instance of Astropy cosmology
        """
        self.astropy_cosmo = astropy_cosmo

    def halo_modifications(self, realization, realization_dict):
        """
        Perform additional modifications on halo properties after process halos
        This method defaults to doing nothing, as it is often not used
        :param realization: an instance of SingleRealization
        :param realization_dict: keyword arguments for the DM model
        :return: realization
        """
        return realization

    def process_halos(self, realization):
        """
        This method takes a given realization and performs operations such as removing very low mass halos far from lensed
        images, cuts on bound mass, etc
        :param realization:
        :return: realization, kwargs_mass_sheet_correction
        """
        raise Exception("This method should be implemented for each user defined class")

    def add_globular_clusters(self, kwargs_globular_clusters):
        """
        This method adds globular clusters
        :param kwargs_globular_clusters:
        :return: return an instance of realization with GCs added
        """
        raise Exception("This method should be implemented for each user defined class")

    def __call__(self, z_lens, z_source, realization_dict):
        """

        :param z_lens: main deflector redshift
        :param z_source: source redshift
        :param realization_dict: a dictionary of keyword arguments
        :return: an instance of SingleRealization in pyHalo
        """
        raise Exception("This method should be implemented for each user defined class")

class ImageMagnificationBaseClass(object):

    def __init__(self):
        pass

    def __call__(self,
                 source_dict,
                 source_x,
                 source_y,
                 astropy_cosmo,
                 z_source):
        """

        :param source_dict: dictionary of keyword arguments for the lensed quasar source
        :param source_x: center of source x coordinate
        :param source_y: center of source y coordinate
        :param astropy_cosmo: an instance of astropy
        :param z_source: source redshift
        :return: magnifications, images, stat, flux_ratios, flux_ratios_data
        """
        raise Exception("This method should be implemented for each user defined class")

class SingleGaussianMagnification(object):
    """Compute the magnification of a lensed quasar image for a quad lens"""

    def __init__(self, astropy_cosmo,
                 rescale_grid_size,
                 rescale_grid_resolution,
                 magnification_method,
                 rotation_angle_list,
                 hessian_eigenvalue_list,
                 fallback='ELLIPTICAL_APERTURE',
                 mu_tolerance=0.05):
        """

        :param astropy_cosmo:
        :param rescale_grid_size:
        :param rescale_grid_resolution:
        :param magnification_method:
        :param rotation_angle_list:
        :param hessian_eigenvalue_list:
        """
        self.astropy_cosmo = astropy_cosmo
        self.rescale_grid_size = rescale_grid_size
        self.rescale_grid_resolution = rescale_grid_resolution
        self.magnification_method = magnification_method
        self.rotation_angle_list = rotation_angle_list
        self.hessian_eigenvalue_list = hessian_eigenvalue_list
        self.fallback = fallback
        self.mu_tolerance = mu_tolerance
        self._mu_discrepancy_list = None

    @property
    def mu_discrepancy_list(self):
        """
        The point-source magnification discrepancy of the near/far splitting approximation for each
        image, or None if the last calculation did not use a near/far splitting method. These are
        the values compared against mu_tolerance to decide whether to fall back on exact ray tracing.
        """
        return self._mu_discrepancy_list

    @property
    def mu_discrepancy(self):
        """
        The largest point-source magnification discrepancy across the images of the last
        calculation. Returns 0 if the last calculation did not use a near/far splitting method.
        """
        if self._mu_discrepancy_list is None:
            return 0.0
        discrepancy = np.array(self._mu_discrepancy_list, dtype=float)
        if np.all(np.isnan(discrepancy)):
            return 0.0
        return float(np.nanmax(discrepancy))

    def grid_resolution(self, source_size):
        return self.rescale_grid_resolution * source_size

    def grid_size_list(self, source_size):

        grid_size_base = auto_raytracing_grid_size(source_size)
        if isinstance(self.rescale_grid_size, list) or isinstance(self.rescale_grid_size, np.ndarray):
            assert len(self.rescale_grid_size) == 4
            grid_size_list = []
            for rescale_size in self.rescale_grid_size:
                grid_size_list.append(rescale_size * grid_size_base)
        else:
            grid_size_list = [self.rescale_grid_size * grid_size_base] * 4
        return grid_size_list

    def __call__(self, source_dict, source_x, source_y, data_class, model_class,
                 lens_model_init, kwargs_lens_init, kwargs_solution,
                 setup_decoupled_multiplane_lens_model_output,
                 lens_model_init_batch, kwargs_lens_init_batch,
                 halo_masses=None,
                 setup_decoupled_multiplane_lens_model_output_batch=None,
                 R_max_grid_size_list=None,
                 verbose=False):
        """
        :param R_max_grid_size_list: optional per-image grid size list used only to set the
            near/far split radius (NEAR_FAR_SPLITTING* methods); defaults to this source's own
            grid_size_list. Lets a caller composing multiple sources (e.g. DoubleGaussianMagnification)
            share a single, larger split radius across all of them.
        """
        source_model_quasar, kwargs_source = setup_gaussian_source(source_dict['source_size_pc'],
                                                                   np.mean(source_x), np.mean(source_y),
                                                                   self.astropy_cosmo, data_class.z_source)
        if isinstance(self.rescale_grid_resolution, list):
            grid_resolution_list = [grid_res_i * auto_raytracing_grid_resolution(
                source_dict['source_size_pc']) for grid_res_i in self.rescale_grid_resolution]
        else:
            grid_resolution_list = [self.rescale_grid_resolution *
                               auto_raytracing_grid_resolution(source_dict['source_size_pc'])] * 4
        grid_size_list = self.grid_size_list(source_dict['source_size_pc'])
        # we pass in setup_decoupled_multiplane_lens_model_output, the decoupled multiplane parameters
        # computed for the proposed macromodel in setup_kwargs_model
        magnifications, images, mu_discrepancy = model_class.image_magnification_gaussian(source_model_quasar,
                                                                              kwargs_source,
                                                                              lens_model_init,
                                                                              kwargs_lens_init,
                                                                              kwargs_solution,
                                                                              grid_size_list,
                                                                              grid_resolution_list,
                                                                              setup_decoupled_multiplane_lens_model_output,
                                                                              magnification_method=self.magnification_method,
                                                                              rotation_angle_list=self.rotation_angle_list,
                                                                              hessian_eigenvalue_list=self.hessian_eigenvalue_list,
                                                                              lens_model_batch=lens_model_init_batch,
                                                                              kwargs_lens_batch=kwargs_lens_init_batch,
                                          setup_decoupled_multiplane_lens_model_output_batch=setup_decoupled_multiplane_lens_model_output_batch,
                                                                              halo_masses=halo_masses,
                                                                              fallback=self.fallback,
                                                                              mu_tolerance=self.mu_tolerance,
                                                                              R_max_grid_size_list=R_max_grid_size_list,
                                                                              verbose=verbose)
        if self.magnification_method in ['NEAR_FAR_SPLITTING', 'NEAR_FAR_SPLITTING_ADAPTIVE']:
            self._mu_discrepancy_list = mu_discrepancy
        else:
            self._mu_discrepancy_list = None
        flux_uncertainty = None
        stat, flux_ratios, flux_ratios_data = flux_ratio_summary_statistic(data_class.magnifications,
                                                                               magnifications,
                                                                                flux_uncertainty,
                                                                               data_class.keep_flux_ratio_index,
                                                                               data_class.uncertainty_in_fluxes)
        return magnifications, images, stat, flux_ratios, flux_ratios_data

class DoubleGaussianMagnification(object):
    """Compute the magnification of a lensed quasar image for a quad lens for two different
    source sizes at the same position in the source plane"""

    def __init__(self, astropy_cosmo,
                 rescale_grid_size,
                 rescale_grid_resolution,
                 magnification_method,
                 rotation_angle_list,
                 hessian_eigenvalue_list,
                 fallback='ELLIPTICAL_APERTURE',
                 mu_tolerance=0.05):
        """

        :param astropy_cosmo:
        :param rescale_grid_size:
        :param rescale_grid_resolution:
        :param magnification_method:
        :param rotation_angle_list:
        :param hessian_eigenvalue_list:
        """
        self.astropy_cosmo = astropy_cosmo
        self.rescale_grid_size = rescale_grid_size
        self.rescale_grid_resolution = rescale_grid_resolution
        self.magnification_method = magnification_method
        self.rotation_angle_list = rotation_angle_list
        self.hessian_eigenvalue_list = hessian_eigenvalue_list
        self.fallback = fallback
        self.mu_tolerance = mu_tolerance
        self._single_source_magnification = SingleGaussianMagnification(astropy_cosmo,
                 rescale_grid_size,
                 rescale_grid_resolution,
                 magnification_method,
                 rotation_angle_list,
                 hessian_eigenvalue_list,
                 fallback=fallback,
                 mu_tolerance=mu_tolerance)

    def __call__(self, source_dict, source_x, source_y, data_class, model_class,
                 lens_model_init, kwargs_lens_init, kwargs_solution,
                 setup_decoupled_multiplane_lens_model_output,
                 lens_model_init_batch, kwargs_lens_init_batch,
                 halo_masses=None,
                 setup_decoupled_multiplane_lens_model_output_batch=None,
                 verbose=False):
        """
        Source 2 defaults to the same source-plane position as source 1. To place source 2 at an
        independent position, include 'source_x_offset_2'/'source_y_offset_2' in kwargs_sample_source
        (any prior type); they default to 0.0 (same position) if not specified.
        """
        source_dict_copy_1 = deepcopy(source_dict)
        source_dict_copy_2 = deepcopy(source_dict)
        source_dict_copy_1['source_size_pc'] = source_dict['source_size_pc_1']
        source_dict_copy_2['source_size_pc'] = source_dict['source_size_pc_2']

        source_x_2 = source_x + source_dict.get('source_x_offset_2', 0.0)
        source_y_2 = source_y + source_dict.get('source_y_offset_2', 0.0)

        # near/far splitting's R_max floor scales with grid size (image_magnification_util.py);
        # use the larger source's grid size for both, so neither source gets a less conservative
        # (more approximate) near/far split than the other.
        grid_size_list_1 = self._single_source_magnification.grid_size_list(source_dict['source_size_pc_1'])
        grid_size_list_2 = self._single_source_magnification.grid_size_list(source_dict['source_size_pc_2'])
        R_max_grid_size_list = list(np.maximum(grid_size_list_1, grid_size_list_2))

        common_args = (data_class, model_class, lens_model_init, kwargs_lens_init, kwargs_solution,
                       setup_decoupled_multiplane_lens_model_output, lens_model_init_batch, kwargs_lens_init_batch)
        common_kwargs = dict(halo_masses=halo_masses,
                             setup_decoupled_multiplane_lens_model_output_batch=setup_decoupled_multiplane_lens_model_output_batch,
                             R_max_grid_size_list=R_max_grid_size_list,
                             verbose=verbose)

        mags_1, images_1, stat_1, flux_ratios_1, flux_ratios_data = self._single_source_magnification(
            source_dict_copy_1, source_x, source_y, *common_args, **common_kwargs)
        mags_2, images_2, stat_2, flux_ratios_2, _ = self._single_source_magnification(
            source_dict_copy_2, source_x_2, source_y_2, *common_args, **common_kwargs)

        magnifications = np.append(mags_1, mags_2)
        images = images_1 + images_2
        stat = np.append(stat_1, stat_2)
        flux_ratios = np.append(flux_ratios_1, flux_ratios_2)
        return magnifications, images, stat, flux_ratios, flux_ratios_data


class PSODoubleGaussianMagnification(object):
    """Like DoubleGaussianMagnification, but source 2's (the larger source) position offset and
    size are PSO-fit against source 2's own flux ratios (data_class.magnifications[4:8]) instead
    of being sampled from a prior every forward-model iteration. Source 1 is unchanged -- still
    the exact lens-equation solution.

    See dual_source_pso_notes.md (repo root) for the full design rationale, and
    samana/source2_position_pso.py for the PSO/near-far machinery this class wires together.

    kwargs_sample_source should still declare 'source_x_offset_2'/'source_y_offset_2'/
    'source_size_pc_2', tagged ['PSO', lo, hi] instead of ['UNIFORM', lo, hi] /
    ['GAUSSIAN', mean, sigma] / etc. -- forward_model_util.sample_prior recognizes the 'PSO' tag
    and skips sampling those 3 keys (they're fit here, via PSO, instead of drawn from a prior
    each iteration), while this class reads the same ['PSO', lo, hi] specs from the same dict as
    its PSO search bounds.
    """

    def __init__(self, astropy_cosmo,
                 rescale_grid_size,
                 rescale_grid_resolution,
                 magnification_method,
                 rotation_angle_list,
                 hessian_eigenvalue_list,
                 kwargs_sample_source,
                 fallback='ELLIPTICAL_APERTURE',
                 mu_tolerance=0.05,
                 n_pso_particles=10,
                 n_pso_iterations=30,
                 pso_seed=None,
                 pso_verbose=False,
                 test_mode=False,
                 max_image_plane_shift=0.01,
                 n_coarse=20,
                 final_n_coarse=40):
        """

        :param astropy_cosmo:
        :param rescale_grid_size:
        :param rescale_grid_resolution:
        :param magnification_method: must be 'NEAR_FAR_SPLITTING_ADAPTIVE' -- the source-2 PSO
            path is only implemented against the adaptive near/far quadtree magnification
            (see mag_finite_single_image_distortion_adaptive_from_splits), so any other method
            here would silently disagree between source 1 (which honors this setting) and
            source 2 (which would not)
        :param rotation_angle_list:
        :param hessian_eigenvalue_list:
        :param kwargs_sample_source: the SAME dict passed to forward_model()'s kwargs_sample_source
            / forward_model_util.sample_prior. Only its 'source_x_offset_2', 'source_y_offset_2',
            'source_size_pc_2' entries are read here, each expected to be a ['PSO', lo, hi] spec
            (converted once, at construction, into PSO search bounds via prior_spec_to_bounds).
            sample_prior recognizes the 'PSO' tag and skips sampling these 3 keys -- their actual
            fitted values come from this class's PSO instead (see last_fit / forward_model_new.py's
            last_fit hook), not from a per-iteration random draw.
        :param n_pso_particles, n_pso_iterations, pso_seed, pso_verbose: source-2 PSO config
        :param test_mode: if True, source 2's returned images carry the winning particle's
            flux_array/tiling (one extra evaluation after the PSO loop finishes -- see
            run_source2_pso's own test_mode param); if False (default), source 2's images are
            [None] * 4, since collecting flux_array/tiling for every PSO particle would be
            wasted work in the common (non-plotting) case.
        :param max_image_plane_shift: passed through to run_source2_pso -- reject any PSO
            candidate whose predicted image-plane shift (for any image) exceeds this (arcsec).
            Default 0.01 (10 mas). See run_source2_pso's own docstring for why this is bounded
            in image-plane, not source-plane, terms.
        :param n_coarse, final_n_coarse: passed through to run_source2_pso. Defaults (20, 40)
            validated against a 20-realization sweep (test_speed_settings_20random*.py, repo
            root, conversation history 2026-08-12): searching at n_coarse=20 vs the production
            n_coarse=40 stayed under ~0.4% max magnification error at the recovered point in
            19/20 cases, but one case reached ~1.2% -- rare but real, matching the discrepancy
            run_source2_pso's own docstring already documents. final_n_coarse=40 buys back that
            outlier risk with a single extra n_coarse=40 evaluation at the winning point (not
            per-particle), so the reported chi2/magnifications are always full-precision even
            though the search itself runs at the cheaper setting.
        """
        if magnification_method != 'NEAR_FAR_SPLITTING_ADAPTIVE':
            raise Exception("PSODoubleGaussianMagnification's source-2 PSO path is only "
                            "implemented against magnification_method='NEAR_FAR_SPLITTING_ADAPTIVE' "
                            "(got '" + str(magnification_method) + "'); source 1 would use a "
                            "different method than source 2's PSO evaluations.")
        self.astropy_cosmo = astropy_cosmo
        self.rescale_grid_size = rescale_grid_size
        self.rescale_grid_resolution = rescale_grid_resolution
        self.magnification_method = magnification_method
        self.rotation_angle_list = rotation_angle_list
        self.hessian_eigenvalue_list = hessian_eigenvalue_list
        self.fallback = fallback
        self.mu_tolerance = mu_tolerance
        self.n_pso_particles = n_pso_particles
        self.n_pso_iterations = n_pso_iterations
        self.pso_seed = pso_seed
        self.pso_verbose = pso_verbose
        self.test_mode = test_mode
        self.max_image_plane_shift = max_image_plane_shift
        self.n_coarse = n_coarse
        self.final_n_coarse = final_n_coarse
        self._single_source_magnification = SingleGaussianMagnification(astropy_cosmo,
                 rescale_grid_size,
                 rescale_grid_resolution,
                 magnification_method,
                 rotation_angle_list,
                 hessian_eigenvalue_list,
                 fallback=fallback,
                 mu_tolerance=mu_tolerance)
        self._offset_x_bounds = prior_spec_to_bounds(kwargs_sample_source['source_x_offset_2'])
        self._offset_y_bounds = prior_spec_to_bounds(kwargs_sample_source['source_y_offset_2'])
        self._size_pc_2_bounds = prior_spec_to_bounds(kwargs_sample_source['source_size_pc_2'])
        # fixed, always-known regardless of whether last_fit has been populated yet -- lets
        # forward_model_new.py's last_fit hook keep a consistent column count in its output
        # even on iterations where this class's __call__ never runs (see
        # split_image_data_reconstruction's magnification-skip branch)
        self.last_fit_param_names = ['source_x_offset_2', 'source_y_offset_2',
                                     'source_size_pc_2', 'source2_pso_chi2', 'source1_chisq']
        self.last_fit = None

    def __call__(self, source_dict, source_x, source_y, data_class, model_class,
                 lens_model_init, kwargs_lens_init, kwargs_solution,
                 setup_decoupled_multiplane_lens_model_output,
                 lens_model_init_batch, kwargs_lens_init_batch,
                 halo_masses=None,
                 setup_decoupled_multiplane_lens_model_output_batch=None,
                 verbose=False):
        source_dict_copy_1 = deepcopy(source_dict)
        source_dict_copy_1['source_size_pc'] = source_dict['source_size_pc_1']

        # source 1: unchanged, exact lens-equation position, normal (un-padded) grid/split
        mags_1, images_1, stat_1, flux_ratios_1, flux_ratios_data = self._single_source_magnification(
            source_dict_copy_1, source_x, source_y, data_class, model_class,
            lens_model_init, kwargs_lens_init, kwargs_solution,
            setup_decoupled_multiplane_lens_model_output,
            lens_model_init_batch, kwargs_lens_init_batch,
            halo_masses=halo_masses,
            setup_decoupled_multiplane_lens_model_output_batch=setup_decoupled_multiplane_lens_model_output_batch,
            verbose=verbose)

        (lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
         z_source, z_split, cosmo_bkg) = setup_decoupled_multiplane_lens_model_output
        if setup_decoupled_multiplane_lens_model_output_batch is not None:
            (lens_model_fixed_batched, _, kwargs_lens_fixed_batched,
             _, _, _, _) = setup_decoupled_multiplane_lens_model_output_batch
        else:
            lens_model_fixed_batched, kwargs_lens_fixed_batched = lens_model_fixed, kwargs_lens_fixed

        # per-image inverse Jacobian -- used to predict, per PSO candidate, where source 2's
        # image actually lands (run_source2_pso rebuilds the near/far split centered there every
        # evaluation; see source2_position_pso.py's module docstring for why a single
        # precomputed/padded split turned out to be inaccurate at nonzero offset). sigma_max is
        # the local source-plane-length -> image-plane-length conversion factor, used below to
        # size the PSO's search bounds dynamically from the current realization's magnification.
        A_inv_list, sigma_max_list = compute_image_hessian_inverse(
            lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
            kwargs_solution, z_split, z_source, cosmo_bkg, data_class.x_image, data_class.y_image)

        # The offset upper bound is a source-plane length scale that gets mapped, via the local
        # Jacobian, into an image-plane footprint -- capped by max_image_plane_shift / sigma_max,
        # using the worst-case (most magnified) of the 4 images since one shared box has to stay
        # safe for all of them. Intersected with (never wider than) the declared prior bounds --
        # this tightens the box on highly-magnified realizations, never loosens it beyond what
        # was declared. FWHM's bound is NOT tied to magnification (tried that, was a mistake --
        # magnification has only ever bounded offset, never size, elsewhere in this design; the
        # per-candidate "offset <= this candidate's own fwhm" check in run_source2_pso already
        # relates the two without constraining fwhm's own range).
        sigma_max_worst = max(sigma_max_list)
        dynamic_max_arcsec = self.max_image_plane_shift / sigma_max_worst

        offset_x_bounds = (max(self._offset_x_bounds[0], -dynamic_max_arcsec),
                          min(self._offset_x_bounds[1], dynamic_max_arcsec))
        offset_y_bounds = (max(self._offset_y_bounds[0], -dynamic_max_arcsec),
                          min(self._offset_y_bounds[1], dynamic_max_arcsec))
        size_pc_2_bounds = self._size_pc_2_bounds

        fit = run_source2_pso(
            data_class, self.astropy_cosmo,
            lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
            kwargs_solution, z_split, z_source, cosmo_bkg,
            lens_model_init_batch, kwargs_lens_init_batch,
            source_x, source_y,
            self.rescale_grid_size, self.rescale_grid_resolution,
            self.rotation_angle_list, self.hessian_eigenvalue_list,
            offset_x_bounds, offset_y_bounds, size_pc_2_bounds,
            A_inv_list, halo_masses=halo_masses,
            lens_model_fixed_batched=lens_model_fixed_batched,
            kwargs_lens_fixed_batched=kwargs_lens_fixed_batched,
            n_particles=self.n_pso_particles, n_iterations=self.n_pso_iterations,
            seed=self.pso_seed, verbose=self.pso_verbose, test_mode=self.test_mode,
            max_image_plane_shift=self.max_image_plane_shift,
            n_coarse=self.n_coarse, final_n_coarse=self.final_n_coarse)

        mags_2 = fit['magnifications']
        stat_2, flux_ratios_2, _ = flux_ratio_summary_statistic(
            np.asarray(data_class.magnifications)[4:8], mags_2, None,
            data_class.keep_flux_ratio_index, data_class.uncertainty_in_fluxes)

        self.last_fit = {'source_x_offset_2': fit['source_x_offset_2'],
                         'source_y_offset_2': fit['source_y_offset_2'],
                         'source_size_pc_2': fit['source_size_pc_2'],
                         'source2_pso_chi2': fit['source2_pso_chi2'],
                         'source1_chisq': stat_1}

        magnifications = np.append(mags_1, mags_2)
        images = images_1 + (fit['images'] if fit['images'] is not None else [None] * len(mags_2))
        # combined (source 1 + source 2) goodness of fit, in quadrature -- same convention used
        # for the chisq-weighted source-2-size histogram/N_eff analysis (conversation history,
        # 2026-08-12). Previously this was np.append(stat_1, stat_2) (a 2-element array), which
        # forward_model_new.py's param_names only had one name for ('summary_statistic'),
        # silently shifting every later output column by one position -- see the
        # forward_model_new.py fix in the same session. A single combined scalar here also makes
        # the acceptance check below (`np.any(stat < tolerance)`, in forward_model_new.py) mean
        # "combined fit is good enough," not "EITHER source's fit alone is good enough" (an OR
        # across two independent stats, which was the accidental behavior of the 2-element form).
        stat = np.sqrt(stat_1 ** 2 + stat_2 ** 2)
        flux_ratios = np.append(flux_ratios_1, flux_ratios_2)
        return magnifications, images, stat, flux_ratios, flux_ratios_data







