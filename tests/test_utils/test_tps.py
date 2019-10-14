import matplotlib.pyplot as plt
import tensorflow as tf
import pytest
import numpy as np

tf.enable_eager_execution()


class Test_Model:
    @pytest.mark.mpl_image_compare
    def test_tps_params(self):
        from eddata.utils.tps import (
            tps_parameters,
            make_input_tps_param,
            ThinPlateSpline,
        )
        from skimage import data

        bs = 1
        scal = 0.8
        tps_scal = 0.05
        rot_scal = 0.1
        off_scal = 0.15
        scal_var = 0.05
        augm_scal = 1.0

        tps_param_dic = tps_parameters(
            2 * bs, scal, tps_scal, rot_scal, off_scal, scal_var, augm_scal
        )

        image = data.astronaut()
        image = np.expand_dims(image, 0)
        image = tf.convert_to_tensor(image)

        orig_images = tf.tile(image, [2, 1, 1, 1])

        coord, vector = make_input_tps_param(tps_param_dic)
        t_images, t_mesh = ThinPlateSpline(orig_images, coord, vector, 128, 3)

        with plt.rc_context({"figure.figsize": [10, 5]}):
            plt.subplot(121)
            plt.imshow(np.squeeze(image[0]))
            plt.subplot(122)
            plt.imshow(np.squeeze(t_images[0]))
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_tps_no_transform_params(self):
        from eddata.utils.tps import (
            make_input_tps_param,
            ThinPlateSpline,
            tps_parameters,
            no_transformation_parameters,
        )
        from skimage import data

        image = data.astronaut()
        image = np.expand_dims(image, 0)
        image = tf.convert_to_tensor(image)

        orig_images = tf.tile(image, [2, 1, 1, 1])

        trf_args = no_transformation_parameters(batch_size=1)
        tps_param_dic = tps_parameters(**trf_args)
        coord, vector = make_input_tps_param(tps_param_dic)
        t_images, t_mesh = ThinPlateSpline(orig_images, coord, vector, 128, 3)

        with plt.rc_context({"figure.figsize": [10, 5]}):
            plt.subplot(121)
            plt.imshow(np.squeeze(image[0]))
            plt.subplot(122)
            plt.imshow(np.squeeze(t_images[0]))
        return plt.gcf()
