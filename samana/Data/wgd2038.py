from samana.Data.data_base import ImagingDataBase
import numpy as np
from samana.Data.ImageData.wgd2038_814w import image_data, psf_error_map, psf_model


class _WGD2038(ImagingDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, supersample_factor=1.0):

        z_lens = 0.23
        z_source = 0.78
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        self._psf_estimate_init = psf_model
        self._psf_error_map_init = psf_error_map
        self._image_data = image_data
        self._supersample_factor = supersample_factor
        image_band = [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]
        multi_band_list = [image_band]
        kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
        likelihood_mask, likelihood_mask_imaging_weights = self.likelihood_masks(x_image, y_image)
        super(_WGD2038, self).__init__(z_lens, z_source,
                                      kwargs_data_joint, x_image, y_image,
                                      magnifications, image_position_uncertainties, flux_uncertainties,
                                      uncertainty_in_fluxes, keep_flux_ratio_index, likelihood_mask,
                                      likelihood_mask_imaging_weights)

    def likelihood_masks(self, x_image, y_image):
        deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size = self.coordinate_properties
        _x = np.linspace(-window_size / 2, window_size / 2, image_data.shape[0])
        _y = np.linspace(-window_size / 2, window_size / 2, image_data.shape[0])
        _xx, _yy = np.meshgrid(_x, _y)
        likelihood_mask = np.ones_like(_xx)
        inds = np.where(np.sqrt(_xx ** 2 + _yy ** 2) >= window_size / 2.)
        likelihood_mask[inds] = 0.0
        return likelihood_mask, likelihood_mask

    @property
    def kwargs_data(self):
        _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
        kwargs_data = {'background_rms': 0.0057,
                           'exposure_time': 1428,
        'ra_at_xy_0': ra_at_xy_0,
        'dec_at_xy_0': dec_at_xy_0,
        'transform_pix2angle': transform_pix2angle,
        'image_data': self._image_data}
        return kwargs_data

    @property
    def kwargs_numerics(self):
        return {'supersampling_factor': int(self._supersample_factor),
                'supersampling_convolution': False}

    @property
    def coordinate_properties(self):
        deltaPix = 0.04
        window_size = 140*deltaPix
        ra_at_xy_0 = 2.8019191
        dec_at_xy_0 = -2.799002
        transform_pix2angle = np.array([[-4.00131855e-02, -1.42308181e-05],
                                        [-1.42320015e-05,  3.99999876e-02]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size

    @property
    def kwargs_psf(self):
        kwargs_psf = {'psf_type': 'PIXEL',
                      'kernel_point_source': self._psf_estimate_init,
                      'psf_error_map': self._psf_error_map_init}
        return kwargs_psf

class WGD2038_HST(_WGD2038):

    def __init__(self, supersample_factor=1.0):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """

        x_image = np.array([-1.474, 0.832, -0.686, 0.706])
        y_image = np.array([0.488, -1.22, -1.191, 0.869])
        magnifications = [0.862069, 1., 0.793103, 0.396552]
        image_position_uncertainties = [0.005]*4
        flux_uncertainties = [0.01, 0.02/1.16, 0.02/0.92, 0.01/0.46] # percent uncertainty
        super(WGD2038_HST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties,
                                         flux_uncertainties,
                                         uncertainty_in_fluxes=True, supersample_factor=supersample_factor)

class WGD2038_JWST(_WGD2038):

    def __init__(self):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        index_reordering = [3,1,2,0]
        x_shift = -0.02
        y_shift = -0.007
        x_image = np.array([ 0.71909739,  0.84809739, -0.66290261, -1.45890261])[index_reordering] + x_shift
        y_image = np.array([ 0.88371611, -1.20728389, -1.17828389,  0.49971611])[index_reordering] + y_shift
        image_position_uncertainties = [0.005] * 4 # 5 marcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)[index_reordering]
        super(WGD2038_JWST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=False)

