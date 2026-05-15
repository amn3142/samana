import numpy as np
import numpy.testing as npt
from samana.analysis_util import quick_setup
from lenstronomy.LensModel.lens_model import LensModel
from samana.image_magnification_util import setup_gaussian_source
from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_resolution, auto_raytracing_grid_size
from lenstronomy.Cosmo.background import Background
from samana.output_storage import Output
from samana.forward_model_util import sample_prior


class TestTwoSources(object):

    def setup_method(self):
        cosmo = Background()
        self._astropy_cosmo = cosmo.cosmo

    def _run_test(self, lens_ID, source_size_pc, rescale_grid_size=1.5, rescale_grid_resolution=2.0):

        data, model = quick_setup(lens_ID)
        data_class = data()
        model_class = model(data_class)

        lens_model_list_macro, redshift_list_macro, _, lens_model_params = model_class.setup_lens_model()
        kwargs_lens_init = lens_model_params[0]
        lens_model = LensModel(lens_model_list_macro,
                               lens_redshift_list=list(redshift_list_macro),
                               multi_plane=True,
                               z_source=data_class.z_source)
        source_x, source_y = lens_model.ray_shooting(data_class.x_image, data_class.y_image, kwargs_lens_init)

        source_model, kwargs_source = setup_gaussian_source(source_size_pc,
                                                            np.mean(source_x), np.mean(source_y),
                                                            self._astropy_cosmo, data_class.z_source)
        grid_size_base = auto_raytracing_grid_size(source_size_pc)
        grid_resolution = rescale_grid_resolution * auto_raytracing_grid_resolution(source_size_pc)
        grid_size_list = [rescale_grid_size * grid_size_base] * len(data_class.x_image)

        magnifications_1, _ = model_class.image_magnification_gaussian(source_model, kwargs_source,
                                                                        lens_model, kwargs_lens_init,
                                                                        kwargs_lens_init,
                                                                        grid_size_list, grid_resolution)
        magnifications_2, _ = model_class.image_magnification_gaussian(source_model, kwargs_source,
                                                                        lens_model, kwargs_lens_init,
                                                                        kwargs_lens_init,
                                                                        grid_size_list, grid_resolution)
        npt.assert_array_equal(magnifications_1, magnifications_2)

    def test_same_source_size_same_magnifications(self):
        self._run_test('J0405', source_size_pc=5.0)

    def test_output_two_sources_property(self):
        mags_1 = np.array([10.0, 8.0, 6.0, 4.0])
        mags_2 = np.array([5.0, 4.0, 3.0, 2.0])
        mags_two_source = np.atleast_2d(np.append(mags_1, mags_2))
        mags_single_source = np.atleast_2d(mags_1)

        out_two = Output(None, mags_two_source, None)
        out_one = Output(None, mags_single_source, None)

        assert out_two.two_sources is True
        assert out_one.two_sources is False

        expected_fr_1 = np.atleast_2d(mags_1[1:] / mags_1[0])
        expected_fr_2 = np.atleast_2d(mags_2[1:] / mags_2[0])
        npt.assert_array_equal(out_two.flux_ratios_1, expected_fr_1)
        npt.assert_array_equal(out_two.flux_ratios_2, expected_fr_2)
        npt.assert_array_equal(out_two.flux_ratios, np.hstack([expected_fr_1, expected_fr_2]))

        try:
            _ = out_one.flux_ratios_2
            raise AssertionError('expected exception not raised')
        except Exception as e:
            assert 'two-source' in str(e)

    def test_sample_prior_two_sources(self):
        kwargs_sample_source = {
            'source_size_pc':   ['UNIFORM', 1.0, 10.0],
            'source_size_pc_2': ['UNIFORM', 1.0, 10.0],
        }
        source_dict, samples, param_names = sample_prior(kwargs_sample_source)

        assert 'source_size_pc' in source_dict
        assert 'source_size_pc_2' in source_dict
        assert 'source_size_pc' in param_names
        assert 'source_size_pc_2' in param_names
        assert len(samples) == 2
        assert 1.0 <= source_dict['source_size_pc'] <= 10.0
        assert 1.0 <= source_dict['source_size_pc_2'] <= 10.0


if __name__ == "__main__":
    t = TestTwoSources()
    t.setup_method()
    t.test_same_source_size_same_magnifications()
    t.test_output_two_sources_property()
    t.test_sample_prior_two_sources()
    print('PASSED')
