#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_affine_grid
"""

import paddle
import pytest
import numpy as np


# places
if paddle.device.is_compiled_with_cuda() is True:
    paddle.device.set_device("gpu:0")
    places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
else:
    paddle.device.set_device("cpu")
    places = [paddle.CPUPlace()]

# types
types = ["float32", "float64"]


def cal_affine_grid_api(theta, out_shape, place, align_corners=True):
    """
    calculate affine_grid api
    """
    theta_p = paddle.to_tensor(theta, stop_gradient=False)
    out_shape_p = paddle.to_tensor(out_shape)
    dynamic_res = paddle.nn.functional.affine_grid(theta_p, out_shape_p, align_corners=align_corners)
    dynamic_res.backward()
    theta_d_grad = theta_p.grad

    paddle.enable_static()
    startup_program, main_program = paddle.static.Program(), paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
            data0 = paddle.static.data(name="s0", shape=theta.shape, dtype=theta.dtype)
            data0.stop_gradient = False
            data1 = paddle.static.data(name="s1", shape=out_shape.shape, dtype="int32")
            feed = {"s0": theta, "s1": out_shape}
            out = paddle.nn.functional.affine_grid(data0, data1, align_corners=align_corners)
            grad = paddle.static.gradients(out, data0)
            exe = paddle.static.Executor(place)
            exe.run(startup_program)
            static_res, theta_s_grad = exe.run(main_program, feed=feed, fetch_list=[out] + [grad])
    paddle.disable_static()
    assert np.allclose(dynamic_res.numpy(), static_res)
    assert np.allclose(theta_d_grad.numpy(), theta_s_grad)

    return static_res


@pytest.mark.api_nn_affine_grid_vartype
def test_affine_grid_base():
    """
    base
    """
    theta = np.array([[[-0.7, -0.4, 0.3], [0.6, 0.5, 1.5]]])
    out_shape = np.array([1, 2, 2, 3], dtype="int32")
    for place in places:
        for dtype in types:
            theta = theta.astype(dtype)
            api_res = cal_affine_grid_api(theta, out_shape, place)
            res = np.array([[[[1.4, 0.4], [0.7, 1.0], [0.0, 1.6]], [[0.6, 1.4], [-0.1, 2.0], [-0.8, 2.6]]]])
            assert np.allclose(api_res, res, atol=1e-5)


@pytest.mark.api_nn_affine_grid_parameter
def test_affine_grid0():
    """
    default
    """
    for place in places:
        np.random.seed(22)
        theta = np.random.rand(4, 2, 3)
        out_shape = np.array([4, 3, 4, 4], dtype="int32")
        api_res = cal_affine_grid_api(theta, out_shape, place)
        res = np.array(
            [
                [
                    [
                        [-0.26960356, -0.69147959],
                        [-0.13062987, -0.11869159],
                        [0.00834382, 0.45409641],
                        [0.14731751, 1.02688441],
                    ],
                    [
                        [0.05151714, -0.57737189],
                        [0.19049084, -0.00458389],
                        [0.32946453, 0.56820411],
                        [0.46843822, 1.14099211],
                    ],
                    [
                        [0.37263785, -0.46326419],
                        [0.51161154, 0.10952381],
                        [0.65058524, 0.68231181],
                        [0.78955893, 1.25509981],
                    ],
                    [
                        [0.69375856, -0.34915648],
                        [0.83273225, 0.22363151],
                        [0.97170594, 0.79641951],
                        [1.11067963, 1.36920751],
                    ],
                ],
                [
                    [
                        [-0.74116967, -0.2612741],
                        [-0.56081444, 0.28002652],
                        [-0.38045922, 0.82132713],
                        [-0.200104, 1.36262774],
                    ],
                    [
                        [-0.28047543, -0.25425618],
                        [-0.10012021, 0.28704443],
                        [0.08023501, 0.82834505],
                        [0.26059023, 1.36964566],
                    ],
                    [
                        [0.1802188, -0.24723827],
                        [0.36057402, 0.29406235],
                        [0.54092924, 0.83536296],
                        [0.72128447, 1.37666358],
                    ],
                    [
                        [0.64091303, -0.24022035],
                        [0.82126826, 0.30108026],
                        [1.00162348, 0.84238088],
                        [1.1819787, 1.38368149],
                    ],
                ],
                [
                    [
                        [-1.36971513, 0.17964743],
                        [-0.827231, 0.18374134],
                        [-0.28474688, 0.18783525],
                        [0.25773725, 0.19192916],
                    ],
                    [
                        [-0.8729816, 0.69434334],
                        [-0.33049747, 0.69843726],
                        [0.21198665, 0.70253117],
                        [0.75447078, 0.70662508],
                    ],
                    [
                        [-0.37624807, 1.20903926],
                        [0.16623606, 1.21313317],
                        [0.70872018, 1.21722708],
                        [1.25120431, 1.22132099],
                    ],
                    [
                        [0.12048546, 1.72373517],
                        [0.66296959, 1.72782908],
                        [1.20545371, 1.73192299],
                        [1.74793784, 1.73601691],
                    ],
                ],
                [
                    [
                        [-0.23152341, -0.46019598],
                        [0.23643518, -0.00138376],
                        [0.70439377, 0.45742845],
                        [1.17235236, 0.91624067],
                    ],
                    [
                        [-0.0331379, -0.20207366],
                        [0.43482069, 0.25673856],
                        [0.90277928, 0.71555077],
                        [1.37073787, 1.17436299],
                    ],
                    [
                        [0.16524761, 0.05604866],
                        [0.6332062, 0.51486088],
                        [1.10116479, 0.97367309],
                        [1.56912338, 1.43248531],
                    ],
                    [
                        [0.36363312, 0.31417098],
                        [0.83159171, 0.7729832],
                        [1.2995503, 1.23179541],
                        [1.76750889, 1.69060763],
                    ],
                ],
            ]
        )
        assert np.allclose(api_res, res)


@pytest.mark.api_nn_affine_grid_parameter
def test_affine_grid1():
    """
    align_corners=False
    """
    for place in places:
        np.random.seed(22)
        theta = np.random.rand(4, 2, 3)
        out_shape = np.array([4, 3, 4, 4], dtype="int32")
        api_res = cal_affine_grid_api(theta, out_shape, place, align_corners=False)
        res = np.array(
            [
                [
                    [
                        [-0.09706816, -0.4338937],
                        [0.0071621, -0.0043027],
                        [0.11139237, 0.4252883],
                        [0.21562264, 0.85487929],
                    ],
                    [
                        [0.14377237, -0.34831293],
                        [0.24800264, 0.08127807],
                        [0.3522329, 0.51086907],
                        [0.45646317, 0.94046007],
                    ],
                    [
                        [0.3846129, -0.26273215],
                        [0.48884317, 0.16685885],
                        [0.59307344, 0.59644985],
                        [0.6973037, 1.02604085],
                    ],
                    [
                        [0.62545343, -0.17715137],
                        [0.7296837, 0.25243963],
                        [0.83391397, 0.68203063],
                        [0.93814423, 1.11162162],
                    ],
                ],
                [
                    [
                        [-0.50077612, -0.05565465],
                        [-0.3655097, 0.35032081],
                        [-0.23024329, 0.75629627],
                        [-0.09497687, 1.16227173],
                    ],
                    [
                        [-0.15525545, -0.05039121],
                        [-0.01998903, 0.35558425],
                        [0.11527739, 0.76155971],
                        [0.2505438, 1.16753517],
                    ],
                    [
                        [0.19026523, -0.04512778],
                        [0.32553165, 0.36084769],
                        [0.46079806, 0.76682315],
                        [0.59606448, 1.17279861],
                    ],
                    [
                        [0.5357859, -0.03986434],
                        [0.67105232, 0.36611112],
                        [0.80631874, 0.77208658],
                        [0.94158515, 1.17806204],
                    ],
                ],
                [
                    [
                        [-0.98000851, 0.37419361],
                        [-0.57314541, 0.37726405],
                        [-0.16628232, 0.38033448],
                        [0.24058077, 0.38340491],
                    ],
                    [
                        [-0.60745836, 0.76021555],
                        [-0.20059527, 0.76328598],
                        [0.20626783, 0.76635642],
                        [0.61313092, 0.76942685],
                    ],
                    [
                        [-0.23490821, 1.14623749],
                        [0.17195488, 1.14930792],
                        [0.57881798, 1.15237835],
                        [0.98568107, 1.15544879],
                    ],
                    [
                        [0.13764194, 1.53225942],
                        [0.54450503, 1.53532986],
                        [0.95136812, 1.53840029],
                        [1.35823122, 1.54147072],
                    ],
                ],
                [
                    [
                        [0.01835563, -0.19134553],
                        [0.36932457, 0.15276363],
                        [0.72029351, 0.4968728],
                        [1.07126245, 0.84098196],
                    ],
                    [
                        [0.16714476, 0.00224621],
                        [0.5181137, 0.34635537],
                        [0.86908265, 0.69046454],
                        [1.22005159, 1.0345737],
                    ],
                    [
                        [0.3159339, 0.19583795],
                        [0.66690284, 0.53994711],
                        [1.01787178, 0.88405628],
                        [1.36884072, 1.22816544],
                    ],
                    [
                        [0.46472303, 0.38942969],
                        [0.81569197, 0.73353885],
                        [1.16666091, 1.07764802],
                        [1.51762986, 1.42175718],
                    ],
                ],
            ]
        )
        assert np.allclose(api_res, res)
