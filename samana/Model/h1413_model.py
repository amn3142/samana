from samana.Model.model_base import EPLModelBase
import numpy as np
from samana.forward_model_util import macromodel_readout_function_eplshear_satellite


class _H1413ModelBase(EPLModelBase):

    @property
    def kwargs_constraints(self):
        joint_source_with_point_source = [[0, 0]]
        kwargs_constraints = {'joint_source_with_point_source': joint_source_with_point_source,
                              'num_point_source_list': [len(self._data.x_image)],
                              'solver_type': 'PROFILE_SHEAR',
                              'point_source_offset': True
                              }
        if self._shapelets_order is not None:
           kwargs_constraints['joint_source_with_source'] = [[0, 1, ['center_x', 'center_y']]]
        return kwargs_constraints

    def setup_source_light_model(self):

        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_init = [
            {'amp': 1, 'R_sersic': 0.02796380070166966, 'n_sersic': 3.9367467127355806, 'e1': 0.10746901603615666,
             'e2': 0.31715572162154915, 'center_x': 0.22411118522966256, 'center_y': 0.49605189911239933}
        ]
        kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1,
                                'center_y': 0.1}]
        kwargs_lower_source = [{'R_sersic': 0.01, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10.0}]
        kwargs_upper_source = [{'R_sersic': 2.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10.0, 'center_y': 10.0}]
        kwargs_source_fixed = [{}]

        if self._shapelets_order is not None:
            n_max = int(self._shapelets_order)
            shapelets_source_list, kwargs_shapelets_init, kwargs_shapelets_sigma, \
            kwargs_shapelets_fixed, kwargs_lower_shapelets, kwargs_upper_shapelets = \
                self.add_shapelets_source(n_max)
            source_model_list += shapelets_source_list
            kwargs_source_init += kwargs_shapelets_init
            kwargs_source_fixed += kwargs_shapelets_fixed
            kwargs_source_sigma += kwargs_shapelets_sigma
            kwargs_lower_source += kwargs_lower_shapelets
            kwargs_upper_source += kwargs_upper_shapelets

        source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_source_fixed, kwargs_lower_source,
                         kwargs_upper_source]

        return source_model_list, source_params

    def setup_lens_light_model(self):

        if self._data.data_band == 'HST814W':
            lens_light_model_list = ['SERSIC']
            kwargs_lens_light_init = [
                {'amp': 1, 'R_sersic': 0.3783137419156091, 'n_sersic': 7.910625151165032,
                 'center_x': -0.0, 'center_y': -0.0}
            ]
            kwargs_lens_light_sigma = [
                {'R_sersic': 0.05, 'n_sersic': 0.25,
                 'center_x': 0.025, 'center_y': 0.025}]
            kwargs_lower_lens_light = [
                {'R_sersic': 0.001, 'n_sersic': 0.5,
                 'center_x': -0.15, 'center_y': -0.15}]
            kwargs_upper_lens_light = [
                {'R_sersic': 5, 'n_sersic': 10.0,
                 'center_x': 0.15, 'center_y': 0.15}]

        else:
            lens_light_model_list = ['SERSIC_ELLIPSE']
            kwargs_lens_light_init = [
                {'amp': 1, 'R_sersic': 0.3783137419156091, 'n_sersic': 7.910625151165032, 'e1': 0.4821410087473546,
                 'e2': 0.09032354893207806, 'center_x': -0.0, 'center_y': -0.0}
            ]
            kwargs_lens_light_sigma = [
                {'R_sersic': 0.05, 'n_sersic': 0.25,
                 'e1': 0.1, 'e2': 0.1,
                 'center_x': 0.025, 'center_y': 0.025}]
            kwargs_lower_lens_light = [
                {'R_sersic': 0.001, 'n_sersic': 0.5,
                 'e1': -0.5, 'e2': -0.5,
                 'center_x': -0.15, 'center_y': -0.15}]
            kwargs_upper_lens_light = [
                {'R_sersic': 5, 'n_sersic': 10.0,
                 'e1': 0.5, 'e2': 0.5,
                 'center_x': 0.15, 'center_y': 0.15}]

        kwargs_lens_light_fixed = [{}]
        lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed, kwargs_lower_lens_light,
                             kwargs_upper_lens_light]
        if self._data.data_band == 'MIRI540W':
            add_uniform_light = True
        else:
            add_uniform_light = False
        if add_uniform_light:
            kwargs_uniform, kwargs_uniform_sigma, kwargs_uniform_fixed, \
                kwargs_uniform_lower, kwargs_uniform_upper = self.add_uniform_lens_light()
            lens_light_model_list += ['UNIFORM']
            kwargs_lens_light_init += kwargs_uniform
            kwargs_lens_light_sigma += kwargs_uniform_sigma
            kwargs_lens_light_fixed += kwargs_uniform_fixed
            kwargs_lower_lens_light += kwargs_uniform_lower
            kwargs_upper_lens_light += kwargs_uniform_upper
        return lens_light_model_list, lens_light_params

    @property
    def kwargs_likelihood(self):
        kwargs_likelihood = {'check_bounds': True,
                             'force_no_add_image': False,
                             'source_marg': False,
                             'image_position_uncertainty': 5e-3,
                             'source_position_tolerance': 0.00001,
                             'source_position_likelihood': True,
                             'prior_lens': self.prior_lens,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True,
                             'custom_logL_addition': self.axis_ratio_masslight_alignment,
                             }
        return kwargs_likelihood

class H1413ModelEPLM3M4Shear(_H1413ModelBase):

    def __init__(self, data_class, shapelets_order=None, shapelets_scale_factor=2.0):
        # shapelets scale factor set to 2; lens model changes with increasing nmax suggesting
        # shapelets are fitting psf noise
        super(H1413ModelEPLM3M4Shear, self).__init__(data_class, shapelets_order, shapelets_scale_factor)

    @property
    def macromodel_readout_function(self):
        return macromodel_readout_function_eplshear_satellite

    @property
    def prior_lens(self):
        return [[2, 'center_x', self._data.g2x, 0.05], [2, 'center_y', self._data.g2y, 0.05], [2, 'theta_E', 0.5, 0.1]]

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR', 'SIS']
        kwargs_lens_macro = [
            {'theta_E': 0.587455262745606, 'gamma': 2.1055423125208983, 'e1': -0.037509215602455695,
             'e2': -0.17957522030983888, 'center_x': 0.009515249579949515, 'center_y': 0.0482884145994108, 'a1_a': 0.0,
             'delta_phi_m1': -0.22390439344439808, 'a3_a': 0.0, 'delta_phi_m3': -0.30191108801054467, 'a4_a': 0.0,
             'delta_phi_m4': -0.23064728686098634},
            {'gamma1': 0.009791375875788436, 'gamma2': -0.1353193461451556, 'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.34489506399633674, 'center_x': 1.4900758399245648, 'center_y': 3.760227228927477}
        ]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens, self._data.z_lens]
        index_lens_split = [0, 1, 2]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a1_a': 0.01, 'delta_phi_m1': 0.1,'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12,
                              'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.1, 'gamma2': 0.1},
                             {'theta_E': 0.2, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0},{}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.6, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi,'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -np.pi/8},
            {'gamma1': -0.5, 'gamma2': -0.5},
        {'theta_E': 0.0, 'center_x': self._data.g2x - 0.25, 'center_y': self._data.g2y - 0.25}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.4, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': np.pi/8},
            {'gamma1': 0.5, 'gamma2': 0.5},
        {'theta_E': 1.2, 'center_x': self._data.g2x + 0.25, 'center_y': self._data.g2y + 0.25}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params
