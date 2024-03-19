import numpy as np
from scipy.stats.kde import gaussian_kde
from scipy.interpolate import interp1d
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.solver4point import Solver4Point
from copy import deepcopy
from samana.Model.multipole_prior import OpticalMultipolePrior


class KwargsLensSampler(object):

    def __init__(self, samples_mcmc, param_class):

        self._samples_mcmc = samples_mcmc
        self._param_class = param_class
        # samples_mcmc shape is (# dims, # data)
        self._kde = gaussian_kde(samples_mcmc.T)

    def draw(self):

        sample = np.squeeze(self._kde.resample(1).T)
        kwargs_lens_init = self._param_class.args2kwargs(sample)['kwargs_lens']
        kwargs_source_init = self._param_class.args2kwargs(sample)['kwargs_source']
        kwargs_lens_light_init = self._param_class.args2kwargs(sample)['kwargs_lens_light']
        return kwargs_lens_init, kwargs_source_init, kwargs_lens_light_init

def sample_prior(kwargs_prior):

    prior_samples_dict = {}
    sample_list = []
    sample_names = []
    joint_multipole_prior_used = False
    for param_name in kwargs_prior.keys():
        if param_name == 'OPTICAL_MULTIPOLE_PRIOR':
            pdf = OpticalMultipolePrior()
            if len(kwargs_prior[param_name]) == 0:
                q_mean, q_sigma = None, None
            else:
                q_mean, q_sigma = kwargs_prior[param_name][0], kwargs_prior[param_name][1]
            a3a, delta_phi_m3, a4a, delta_phi_m4 = pdf.sample(q_mean, q_sigma)
            prior_samples_dict['a3_a'] = a3a
            prior_samples_dict['a4_a'] = a4a
            prior_samples_dict['delta_phi_m3'] = delta_phi_m3
            prior_samples_dict['delta_phi_m4'] = delta_phi_m4
            sample_list += [a3a, a4a, delta_phi_m3, delta_phi_m4]
            sample_names += ['a3_a', 'a4_a', 'delta_phi_m3', 'delta_phi_m4']
            joint_multipole_prior_used = True
        elif param_name == 'BAYESIAN_HIERARCHICAL_MULTIPOLES':
            raise Exception('not yet implemented')
        else:
            prior_type = kwargs_prior[param_name][0]
            if prior_type == 'FIXED':
                sample = kwargs_prior[param_name][1]
                prior_samples_dict[param_name] = sample
                continue
            elif prior_type == 'UNIFORM':
                param_min, param_max = kwargs_prior[param_name][1], kwargs_prior[param_name][2]
                sample = np.random.uniform(param_min, param_max)
            elif prior_type == 'GAUSSIAN':
                mean, standard_dev = kwargs_prior[param_name][1], kwargs_prior[param_name][2]
                sample = np.random.normal(mean, standard_dev)
            else:
                raise Exception('only UNIFORM, GAUSSIAN, and FIXED priors currently implemented')
            if joint_multipole_prior_used and param_name in ['a3_a', 'a4_a', 'delta_phi_m3', 'delta_phi_m3']:
                raise Exception('you have specified a prior on multipole moments separately '
                                 'from JOINT_MULTIPOLE_PRIOR, which is not currently allowed!')
            prior_samples_dict[param_name] = sample
            sample_list.append(sample)
            sample_names.append(param_name)

    return prior_samples_dict, np.array(sample_list), sample_names

def filenames(output_path, job_index):
    """
    Creates the names for output files in a certain format
    :param output_path: the directly where output will be produced; individual jobs (indexed by job_index) will be created
    in directories output_path/job_1, output_path/job_2, etc. where the 1, 2 are set by job_index
    :param job_index: a unique integer that specifies the output folder number
    :return: the output filenames
    """
    filename_parameters = output_path + 'job_' + str(job_index) + '/parameters.txt'
    filename_mags = output_path + 'job_' + str(job_index) + '/fluxes.txt'
    filename_realizations = output_path + 'job_' + str(job_index) + '/'
    filename_sampling_rate = output_path + 'job_' + str(job_index) + '/sampling_rate.txt'
    filename_acceptance_ratio = output_path + 'job_' + str(job_index) + '/acceptance_ratio.txt'
    filename_macromodel_samples = output_path + 'job_' + str(job_index) + '/macromodel_samples.txt'
    return filename_parameters, filename_mags, filename_realizations, filename_sampling_rate, filename_acceptance_ratio, \
           filename_macromodel_samples

def ray_angles(alpha_x, alpha_y, lens_model, kwargs_lens, zsource):

    redshift_list = lens_model.redshift_list + [zsource]
    redshift_list_finely_sampled = np.arange(0.02, zsource, 0.02)
    full_redshift_list = np.unique(np.append(redshift_list, redshift_list_finely_sampled))
    full_redshift_list_sorted = full_redshift_list[np.argsort(full_redshift_list)]
    x_angle_list, y_angle_list, tz = [alpha_x], [alpha_y], [0.]

    cosmo_calc = lens_model.lens_model._multi_plane_base._cosmo_bkg.T_xy

    x0, y0 = 0., 0.
    zstart = 0.
    for zi in full_redshift_list_sorted:
        assert len(lens_model.lens_model_list) == len(kwargs_lens)
        x0, y0, alpha_x, alpha_y = lens_model.lens_model.ray_shooting_partial(x0, y0, alpha_x, alpha_y, zstart, zi,
                                                                              kwargs_lens)
        d = cosmo_calc(0., zi)
        x_angle_list.append(x0 / d)
        y_angle_list.append(y0 / d)
        tz.append(d)
        zstart = zi
        # if hasattr(lens_model, 'lens_model'):
        #     x0, y0, alpha_x, alpha_y = lens_model.lens_model.ray_shooting_partial(x0, y0, alpha_x, alpha_y, zstart, zi,
        #                                                                           kwargs_lens)
        #     d = cosmo_calc(0., zi)
        # elif hasattr(lens_model, 'ray_shooting_partial'):
        #     x0, y0, alpha_x, alpha_y = lens_model.ray_shooting_partial(x0, y0, alpha_x, alpha_y, zstart, zi,
        #                                                                kwargs_lens)
        #     d = cosmo_calc(zi).value
        # else:
        #     raise Exception('the supplied lens model class does not have a ray shooting partial method')
    return x_angle_list, y_angle_list, tz

def interpolate_ray_paths(x_image, y_image, lens_model, kwargs_lens, zsource,
                          terminate_at_source=False, source_x=None, source_y=None):
    """
    :param x_image: x coordinates to interpolate (arcsec)
    :param y_image: y coordinates to interpolate (arcsec)
    :param lens_model: instance of LensModel
    :param kwargs_lens: keyword arguments for lens model
    :param zsource: source redshift
    :param terminate_at_source: fix the final angular coordinate to the source coordinate
    :param source_x: source x coordinate (arcsec)
    :param source_y: source y coordinate (arcsec)
    :return: Instances of interp1d (scipy) that return the angular coordinate of a ray given a
    comoving distance
    """

    ray_angles_x = []
    ray_angles_y = []

    # print('coordinate: ', (x_image, y_image))
    for (xi, yi) in zip(x_image, y_image):

        angle_x, angle_y, tz = ray_angles(xi, yi, lens_model, kwargs_lens, zsource)

        if terminate_at_source:
            angle_x[-1] = source_x
            angle_y[-1] = source_y

        ray_angles_x.append(interp1d(tz, angle_x))
        ray_angles_y.append(interp1d(tz, angle_y))

    return ray_angles_x, ray_angles_y

def align_realization(realization, lens_model_list_macro, redshift_list_macro, kwargs_lens_init, x_image, y_image):
    """

    :param realization:
    :param lens_model_list_macro:
    :param redshift_list_macro:
    :param kwargs_lens_init:
    :param x_image:
    :param y_image:
    :return:
    """
    z_source = realization.lens_cosmo.z_source
    cosmo = realization.lens_cosmo.cosmo.astropy
    lens_model = LensModel(lens_model_list_macro, lens_redshift_list=redshift_list_macro,
                           z_source=z_source, multi_plane=True, cosmo=cosmo)

    solver = Solver4Point(lens_model, solver_type='PROFILE_SHEAR')
    kwargs_lens, _ = solver.constraint_lensmodel(x_image, y_image, kwargs_lens_init)
    source_x, source_y = lens_model.ray_shooting(x_image[0], y_image[0], kwargs_lens)
    ray_interp_x, ray_interp_y = interpolate_ray_paths(
        x_image, y_image, lens_model, kwargs_lens, z_source, terminate_at_source=True,
        source_x=source_x, source_y=source_y)
    ### Now compute the centroid of the light cone as the coordinate centroid of the individual images
    z_range = np.linspace(0, z_source, 100)
    distances = [realization.lens_cosmo.cosmo.D_C_transverse(zi) for zi in z_range]
    angular_coordinates_x = []
    angular_coordinates_y = []
    for di in distances:
        x_coords = [ray_x(di) for i, ray_x in enumerate(ray_interp_x)]
        y_coords = [ray_y(di) for i, ray_y in enumerate(ray_interp_y)]
        x_center = np.mean(x_coords)
        y_center = np.mean(y_coords)
        angular_coordinates_x.append(x_center)
        angular_coordinates_y.append(y_center)
    ray_interp_x = [interp1d(distances, angular_coordinates_x)]
    ray_interp_y = [interp1d(distances, angular_coordinates_y)]
    realization = realization.shift_background_to_source(ray_interp_x[0], ray_interp_y[0])
    return realization, ray_interp_x, ray_interp_y, lens_model, kwargs_lens

def flux_ratio_summary_statistic(normalized_magnifcations_measured, model_magnifications,
                          measurement_uncertainties, keep_flux_ratio_index, uncertainty_in_fluxes):
    """
    Computes the summary statistic corresponding to a set of flux ratios
    :param normalized_magnifcations_measured:
    :param model_magnifications:
    :param uncertainty_in_fluxes:
    :param magnification_uncertainties:
    :param keep_flux_ratio_index:
    :return:
    """
    _flux_ratios_data = np.array(normalized_magnifcations_measured[1:]) / normalized_magnifcations_measured[0]
    # account for measurement uncertainties in the measured fluxes or flux ratios
    if uncertainty_in_fluxes:
        if measurement_uncertainties is None:
            mags_with_uncertainties = np.deepcopy(model_magnifications)
        else:
            assert len(measurement_uncertainties) == len(model_magnifications)
            mags_with_uncertainties = [model_magnifications[j] +
                                       np.random.normal(0.0, measurement_uncertainties[j]*model_magnifications[j])
                                       for j in range(0, len(model_magnifications))]
        _flux_ratios = np.array(mags_with_uncertainties)[1:] / mags_with_uncertainties[0]
    else:
        _fr = model_magnifications[1:] / model_magnifications[0]
        if measurement_uncertainties is None:
            fluxratios_with_uncertainties = deepcopy(_fr)
        else:
            assert len(measurement_uncertainties) == len(model_magnifications) - 1
            fluxratios_with_uncertainties = [_fr[j] + np.random.normal(0.0, _fr[j]*measurement_uncertainties[j])
                                             for j in range(0, len(_fr))]
        _flux_ratios = np.array(fluxratios_with_uncertainties)
    flux_ratios_data = []
    flux_ratios = []
    for idx in keep_flux_ratio_index:
        flux_ratios.append(_flux_ratios[idx])
        flux_ratios_data.append(_flux_ratios_data[idx])
    # Now we compute the summary statistic
    stat = 0
    for f_i_data, f_i_model in zip(flux_ratios_data, flux_ratios):
        stat += (f_i_data - f_i_model) ** 2
    stat = np.sqrt(stat)
    return stat, flux_ratios, flux_ratios_data

def flux_ratio_likelihood(measured_fluxes, model_fluxes, measurement_uncertainties, uncertainty_in_fluxes,
                          keep_flux_ratio_index, tolerance=0.03):
    """

    :param measured_fluxes:
    :param model_fluxes:
    :param measurement_uncertainties:
    :param uncertainty_in_fluxes:
    :param keep_flux_ratio_index:
    :param tolerance:
    :return:
    """
    if uncertainty_in_fluxes:
        n_draw = 500000
        measured_flux_ratios = measured_fluxes[1:] / measured_fluxes[0]
        if measurement_uncertainties is None:
            _model_flux = deepcopy(model_fluxes)
        else:
            _model_flux = np.random.normal(model_fluxes, measurement_uncertainties, size=(n_draw, 4))
        model_flux_ratios = _model_flux[:, 1:] / _model_flux[:, 0, np.newaxis]
        delta = 0
        for i in range(0, 3):
            if i in keep_flux_ratio_index:
                delta += (measured_flux_ratios[i] - model_flux_ratios[:, i]) ** 2
        delta = np.sqrt(delta)
        importance_weight = np.sum(delta <= tolerance) / n_draw
        return importance_weight

    else:
        model_flux_ratios = model_fluxes[1:] / model_fluxes[0]
        measured_flux_ratios = measured_fluxes[1:] / measured_fluxes[0]
        importance_weight = 0.0
        for i in range(0, 3):
            if i not in keep_flux_ratio_index:
                continue
            if measurement_uncertainties is None:
                # pick a fiducial uncertainty; all we really need is the relative likelihood anyways
                df = (model_flux_ratios[i] - measured_flux_ratios[i]) / 0.01
            else:
                df = (model_flux_ratios[i] - measured_flux_ratios[i]) / measurement_uncertainties[i]
            importance_weight += df ** 2
        return np.exp(-0.5 * importance_weight)

def check_solution(source_x, source_y, tolerance=0.0001):
    """
    Verifies the degree to which the lens equation is satisfied based on the scatter in the source position
    :param source_x: source coordinate
    :param source_y: source coordiante
    :param tolerance: tolerance applied in the source plane
    :return: metric
    """
    num_images = len(source_x)
    penalty = 0
    if num_images == 4:
        penalty += (source_x[0] - source_x[1]) ** 2
        penalty += (source_x[0] - source_x[2]) ** 2
        penalty += (source_x[0] - source_x[3]) ** 2
        penalty += (source_x[1] - source_x[2]) ** 2
        penalty += (source_x[1] - source_x[3]) ** 2
        penalty += (source_x[2] - source_x[3]) ** 2
        penalty += (source_y[0] - source_y[1]) ** 2
        penalty += (source_y[0] - source_y[2]) ** 2
        penalty += (source_y[0] - source_y[3]) ** 2
        penalty += (source_y[1] - source_y[2]) ** 2
        penalty += (source_y[1] - source_y[3]) ** 2
        penalty += (source_y[2] - source_y[3]) ** 2
    elif num_images == 2:
        penalty += (source_x[0] - source_x[1]) ** 2
        penalty += (source_y[0] - source_y[1]) ** 2
    else:
        raise Exception('check solution only implemented for 2 and 4 images')
    return np.sqrt(penalty) / tolerance / num_images
