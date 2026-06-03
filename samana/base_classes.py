from samana.image_magnification_util import setup_gaussian_source, source_optimizer_pso, _build_tnfw_groups
from samana.forward_model_util import flux_ratio_summary_statistic
from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_size, auto_raytracing_grid_resolution

import numpy as np


class DarkMatterBaseClass(object):

    def __init__(self, astropy_cosmo):
        """
        astropy_cosmo: an instance of Astropy cosmology
        """
        self.astropy_cosmo = astropy_cosmo

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
                 data_class,
                 model_class,
                 lens_model_init,
                 kwargs_lens_init,
                 kwargs_solution,
                 setup_decoupled_multiplane_lens_model_output,
                 index_lens_split,
                 seed=None):
        """

        :param source_dict: dictionary of keyword arguments for the lensed quasar source
        :param source_x: center of source x coordinate
        :param source_y: center of source y coordinate
        :param astropy_cosmo: an instance of astropy
        :param data_class: data class instance
        :param model_class: model class instance
        :param lens_model_init: full lenstronomy LensModel
        :param kwargs_lens_init: full lens kwargs
        :param kwargs_solution: fitted macromodel kwargs
        :param setup_decoupled_multiplane_lens_model_output: precomputed decoupled multiplane output
        :param index_lens_split: indices of macromodel components in kwargs_lens_init
        :param seed: random seed for reproducibility
        :return: magnifications, images, stat, flux_ratios, flux_ratios_data, optimizer_output
        """
        raise Exception("This method should be implemented for each user defined class")

class SingleGaussianMagnification(object):
    """Compute the magnification of a lensed quasar image for a quad lens"""

    def __init__(self, astropy_cosmo,
                 rescale_grid_size,
                 rescale_grid_resolution,
                 magnification_method,
                 rotation_angle_list,
                 hessian_eigenvalue_list):
        self.astropy_cosmo = astropy_cosmo
        self.rescale_grid_size = rescale_grid_size
        self.rescale_grid_resolution = rescale_grid_resolution
        self.magnification_method = magnification_method
        self.rotation_angle_list = rotation_angle_list
        self.hessian_eigenvalue_list = hessian_eigenvalue_list

    def __call__(self, source_dict, source_x, source_y, astropy_cosmo, data_class, model_class,
                 lens_model_init, kwargs_lens_init, kwargs_solution, setup_decoupled_multiplane_lens_model_output,
                 index_lens_split, seed=None):

        source_model_quasar, kwargs_source = setup_gaussian_source(source_dict['source_size_pc'],
                                                                   np.mean(source_x), np.mean(source_y),
                                                                   astropy_cosmo, data_class.z_source)
        grid_size_base = auto_raytracing_grid_size(source_dict['source_size_pc'])
        grid_resolution = self.rescale_grid_resolution * auto_raytracing_grid_resolution(source_dict['source_size_pc'])
        if isinstance(self.rescale_grid_size, list) or isinstance(self.rescale_grid_size, np.ndarray):
            assert len(self.rescale_grid_size) == 4
            grid_size_list = []
            for rescale_size in self.rescale_grid_size:
                grid_size_list.append(rescale_size * grid_size_base)
        else:
            grid_size_list = [self.rescale_grid_size * grid_size_base] * 4
        magnifications, images = model_class.image_magnification_gaussian(source_model_quasar,
                                                                              kwargs_source,
                                                                              lens_model_init,
                                                                              kwargs_lens_init,
                                                                              kwargs_solution,
                                                                              grid_size_list,
                                                                              grid_resolution,
                                                                              setup_decoupled_multiplane_lens_model_output,
                                                                              magnification_method=self.magnification_method,
                                                                              rotation_angle_list=self.rotation_angle_list,
                                                                              hessian_eigenvalue_list=self.hessian_eigenvalue_list)
        flux_uncertainty = None
        stat, flux_ratios, flux_ratios_data = flux_ratio_summary_statistic(data_class.magnifications,
                                                                               magnifications,
                                                                                flux_uncertainty,
                                                                               data_class.keep_flux_ratio_index,
                                                                               data_class.uncertainty_in_fluxes)
        return magnifications, images, stat, flux_ratios, flux_ratios_data, None


class SourceOptimizerMagnification(object):
    """Compute magnifications by optimizing source position and size with a PSO."""

    def __init__(self, rescale_grid_size, rescale_grid_resolution, **kwargs_pso):
        """
        :param rescale_grid_size: grid size multiplier passed to source_optimizer_pso
        :param rescale_grid_resolution: grid resolution multiplier passed to source_optimizer_pso
        :param kwargs_pso: any additional keyword arguments forwarded to source_optimizer_pso,
            e.g. n_particles, n_iterations, use_vectorized_ray_shooting, verbose
        """
        self.rescale_grid_size = rescale_grid_size
        self.rescale_grid_resolution = rescale_grid_resolution
        self.kwargs_pso = kwargs_pso

    def __call__(self, source_dict, source_x, source_y, astropy_cosmo, data_class, model_class,
                 lens_model_init, kwargs_lens_init, kwargs_solution, setup_decoupled_multiplane_lens_model_output,
                 index_lens_split, seed=None):

        use_vectorized = self.kwargs_pso.get('use_vectorized_ray_shooting', True)
        if setup_decoupled_multiplane_lens_model_output is not None and use_vectorized:
            lmf, _, kwf, _, _, _, _ = setup_decoupled_multiplane_lens_model_output
            groups = _build_tnfw_groups(lmf, kwf)
        else:
            groups = None

        magnifications, images, stat, optimizer_output = source_optimizer_pso(
            lens_model_init, kwargs_lens_init, kwargs_solution, index_lens_split,
            data_class.x_image, data_class.y_image,
            float(np.mean(source_x)), float(np.mean(source_y)),
            source_dict['source_size_pc'],
            data_class.magnifications, data_class.keep_flux_ratio_index,
            data_class.z_source, astropy_cosmo,
            magnification_method='ELLIPTICAL_APERTURE',
            rescale_grid_size=self.rescale_grid_size,
            rescale_grid_resolution=self.rescale_grid_resolution,
            flux_ratio_covariance_matrix=data_class.flux_ratio_covariance_matrix,
            setup_decoupled_multiplane_lens_model_output=setup_decoupled_multiplane_lens_model_output,
            groups=groups,
            seed=seed,
            **self.kwargs_pso,
        )

        if magnifications is None:
            return None, None, np.inf, None, None, None

        flux_ratios = list(magnifications[1:] / magnifications[0])
        flux_ratios_data = list(np.array(data_class.magnifications[1:]) / data_class.magnifications[0])

        return magnifications, images, stat, flux_ratios, flux_ratios_data, optimizer_output
