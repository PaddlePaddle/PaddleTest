#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.AdaptiveAvgPool3D
"""
import copy

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np

_dtype = np.float32


class TestRNN(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        self.seed = 100
        # self.enable_backward=False
        self.delta = 0.0001
        self.forward_kwargs = {}  # 前向传播参数

    def _static_forward(self, res, data=None, **kwargs):
        """
        _static_forward
        """

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        main_program.random_seed = self.seed

        with paddle.utils.unique_name.guard():
            with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                if data is not None:
                    data = data.astype(self.dtype)
                    self.data = paddle.static.data(name="data", shape=data.shape, dtype=self.dtype)
                    self.data.stop_gradient = False
                data = dict({"data": data}, **kwargs)
                static_cell = paddle.nn.SimpleRNNCell(4, 8)
                parameters = {}
                for k, v in kwargs["cell"].named_parameters():
                    parameters[k] = v

                obj = self.func(
                    static_cell,
                    is_reverse=self.kwargs.get("is_reverse", False),
                    time_major=self.kwargs.get("time_major", False),
                )
                output, h = obj(
                    self.data,
                    initial_states=self.forward_kwargs.get("initial_states", None),
                    sequence_length=self.forward_kwargs.get("sequence_length", None),
                )

                if self.enable_backward:
                    loss = paddle.mean(output)
                    g = paddle.static.gradients(loss, self.data)
                    exe = paddle.static.Executor(self.place)
                    exe.run(startup_program)
                    for k, v in static_cell.named_parameters():
                        v.set_value(parameters[k])
                    res = exe.run(main_program, feed=data, fetch_list=[output, h, g], return_numpy=True)
                    grad = {"data": res[2]}
                    return res[0:2], grad
                else:
                    exe = paddle.static.Executor(self.place)
                    exe.run(startup_program)
                    for k, v in static_cell.named_parameters():
                        v.set_value(parameters[k])
                    res = exe.run(main_program, feed=data, fetch_list=[output, h], return_numpy=True)
                    return res

    def _dygraph_forward(self):
        """
        _dygraph_forward
        """
        cell = copy.deepcopy(self.kwargs.get("cell"))
        obj = self.func(
            cell=cell, is_reverse=self.kwargs.get("is_reverse", False), time_major=self.kwargs.get("time_major", False)
        )
        res = obj(
            self.data,
            initial_states=self.forward_kwargs.get("initial_states", None),
            sequence_length=self.forward_kwargs.get("sequence_length", None),
        )
        return res


obj = TestRNN(paddle.nn.RNN)
res_outputs = [
    [
        [
            -0.41805826338051666,
            0.22022120468076872,
            -0.338273189723404,
            -0.24617478149067853,
            0.09573324424067524,
            -0.47079110200581753,
            -0.03435668862995602,
            -0.40527382273501933,
        ],
        [
            -0.16655437807618048,
            0.2769589075445124,
            -0.14167845285320205,
            -0.47032005200197013,
            0.1028938747907291,
            -0.2342414953727308,
            0.21773937893402742,
            -0.33428361063120415,
        ],
        [
            -0.30962178642274835,
            0.14995033887532402,
            -0.5014789422897543,
            -0.5575169371542475,
            0.3508976438664536,
            -0.14612803096924393,
            0.055716862608673354,
            -0.23707090722122118,
        ],
    ],
    [
        [
            -0.2910432922599807,
            0.20006867702342945,
            -0.13309779241132294,
            -0.35354838575117636,
            -0.0382390069374574,
            -0.46651599812797767,
            -0.0548339255988763,
            -0.3802476536920616,
        ],
        [
            -0.379896702128952,
            0.440035048064823,
            -0.5127017093856834,
            -0.2530757972388444,
            0.4495343467088128,
            -0.3693239736103003,
            -0.04066560667765106,
            -0.3609407859877686,
        ],
        [
            -0.32279289579646764,
            0.07392642831034953,
            -0.42972779798163085,
            -0.6088280646556109,
            0.320803940263583,
            -0.05130848141305536,
            0.3858006307601003,
            -0.1703522265261066,
        ],
    ],
]
res_final_test = [
    [
        -0.30962178642274835,
        0.14995033887532402,
        -0.5014789422897543,
        -0.5575169371542475,
        0.3508976438664536,
        -0.14612803096924393,
        0.055716862608673354,
        -0.23707090722122118,
    ],
    [
        -0.32279289579646764,
        0.07392642831034953,
        -0.42972779798163085,
        -0.6088280646556109,
        0.320803940263583,
        -0.05130848141305536,
        0.3858006307601003,
        -0.1703522265261066,
    ],
]
np_cell_params = {
    "weight_ih": np.array(
        [
            [-0.22951947, -0.08992132, -0.34953101, -0.175061],
            [0.20906496, -0.3427665, 0.06989282, 0.07340089],
            [-0.27920275, -0.08347859, -0.32776092, 0.27606266],
            [0.3400624, -0.311168, 0.27615769, 0.05437757],
            [0.17145903, 0.09205394, 0.05787117, -0.33910074],
            [-0.20504217, 0.03159698, 0.19029316, -0.17628509],
            [-0.15139461, 0.24918096, 0.33588031, 0.27213237],
            [-0.09934296, 0.06990383, -0.10267501, -0.11300258],
        ],
        dtype=_dtype,
    ),
    "weight_hh": np.array(
        [
            [-0.22763112, -0.1854782, -0.32183097, 0.0038406, -0.08750273, 0.06562333, 0.09188278, -0.25271974],
            [0.30677212, 0.31563824, 0.07233466, -0.07936122, -0.09674069, -0.20905946, -0.15785094, -0.1792262],
            [-0.230794, 0.32994288, 0.32315671, 0.06927786, 0.16355433, -0.11286469, -0.28846025, -0.0258108],
            [0.00615105, -0.2910026, 0.0198239, 0.34800829, -0.0742208, -0.11625087, 0.21598615, 0.1798519],
            [-0.13218199, 0.09477825, 0.02857035, -0.14368852, -0.27521451, -0.13248332, -0.03042035, 0.1123876],
            [-0.17376618, 0.09977366, -0.21204463, 0.11145757, 0.19678019, 0.19770592, 0.07801379, -0.13505715],
            [0.13981969, 0.25428854, 0.08861728, 0.34111385, 0.33693647, -0.23568284, -0.33716397, -0.23988983],
            [0.29945748, 0.32070817, -0.20436912, -0.09862354, 0.03491358, -0.16133995, -0.02785886, 0.13870717],
        ],
        dtype=_dtype,
    ),
    "bias_ih": np.array(
        [
            2.51656952e-04,
            1.52785263e-01,
            1.83536185e-02,
            -3.52564132e-01,
            -7.44581413e-02,
            -5.53878870e-03,
            -6.86739763e-02,
            -1.03026660e-01,
        ],
        dtype=_dtype,
    ),
    "bias_hh": np.array(
        [0.00043439, -0.03876598, -0.28960775, -0.16011519, 0.31358566, -0.33478349, -0.32527005, -0.15334292],
        dtype=_dtype,
    ),
}


@pytest.mark.api_nn_RNN_vartype
def test_rnn_base():
    """
    base
    """
    np.random.seed(obj.seed)
    inputs = randtool("float", 0, 1, (2, 3, 4))

    # prev_h = randtool("float", 0, 1, (2, 8))
    cell = paddle.nn.SimpleRNNCell(4, 8)
    # set the weights
    for k, v in cell.named_parameters():
        v.set_value(np_cell_params[k])

    res = [res_outputs, res_final_test]
    obj.base(res, data=inputs, cell=cell)


@pytest.mark.api_nn_RNN_parameters
def test_rnn1():
    """
    default
    """
    np.random.seed(obj.seed)
    inputs = randtool("float", 0, 1, (2, 3, 4))

    # prev_h = randtool("float", 0, 1, (2, 8))
    cell = paddle.nn.SimpleRNNCell(4, 8)
    # set the weights
    for k, v in cell.named_parameters():
        v.set_value(np_cell_params[k])

    res = [res_outputs, res_final_test]
    obj.run(res, data=inputs, cell=cell)


#
@pytest.mark.apt_nn_RNN_parameters
def test_rnn2():
    """
    set is_reverse=True
    """
    np.random.seed(obj.seed)
    inputs = randtool("float", 0, 1, (2, 3, 4))
    # prev_h = randtool("float", 0, 1, (2, 8))
    cell = paddle.nn.SimpleRNNCell(4, 8)
    # set the weights
    for k, v in cell.named_parameters():
        v.set_value(np_cell_params[k])

    res_outputs = [
        [
            [
                -0.2589979759048145,
                0.3240111769424469,
                -0.3930827679031371,
                -0.4550450697483109,
                0.14750135709874887,
                -0.4174765422774691,
                -0.09569240111718305,
                -0.3245812517782133,
            ],
            [
                -0.08013370039343207,
                0.1170042730339912,
                -0.30515483626503426,
                -0.46595279283909635,
                0.012902156875043688,
                -0.1443459732860601,
                0.060135292018463474,
                -0.36570477654937,
            ],
            [
                -0.40579899720523244,
                0.02312562316595211,
                -0.5312022907527016,
                -0.3693065415065509,
                0.28778352515924405,
                -0.21408476966018827,
                0.08476590823896107,
                -0.331850261752073,
            ],
        ],
        [
            [
                -0.04323715734876021,
                0.2758733339839664,
                -0.11169581261110856,
                -0.5577216056359302,
                -0.09549122438928458,
                -0.26012886317201583,
                0.15354536934902546,
                -0.24630044505196771,
            ],
            [
                -0.1823227115460985,
                0.19892074311953015,
                -0.6418025254265327,
                -0.20776887808392297,
                0.34250159108287603,
                -0.17313980368980414,
                -0.1723277295300889,
                -0.4085417079960605,
            ],
            [
                -0.48351457743645987,
                -0.03610278540308465,
                -0.5686091525998158,
                -0.3921202383856199,
                0.32123451854395846,
                -0.29386941479433254,
                0.1401073686084718,
                -0.34122503415683153,
            ],
        ],
    ]

    res_final_test = [
        [
            -0.2589979759048145,
            0.3240111769424469,
            -0.3930827679031371,
            -0.4550450697483109,
            0.14750135709874887,
            -0.4174765422774691,
            -0.09569240111718305,
            -0.3245812517782133,
        ],
        [
            -0.04323715734876021,
            0.2758733339839664,
            -0.11169581261110856,
            -0.5577216056359302,
            -0.09549122438928458,
            -0.26012886317201583,
            0.15354536934902546,
            -0.24630044505196771,
        ],
    ]

    res = [res_outputs, res_final_test]
    obj.run(res, data=inputs, cell=cell, is_reverse=True)


@pytest.mark.apt_nn_RNN_parameters
def test_rnn3():
    """
    set time_major = True
    """
    np.random.seed(obj.seed)
    inputs = randtool("float", 0, 1, (2, 3, 4))

    # prev_h = randtool("float", 0, 1, (2, 8))
    cell = paddle.nn.SimpleRNNCell(4, 8)
    # set the weights
    for k, v in cell.named_parameters():
        v.set_value(np_cell_params[k])
    res_outputs = [
        [
            [
                -0.41805826338051666,
                0.22022120468076872,
                -0.338273189723404,
                -0.24617478149067853,
                0.09573324424067524,
                -0.47079110200581753,
                -0.03435668862995602,
                -0.40527382273501933,
            ],
            [
                -0.3716626927858596,
                0.178889158651149,
                -0.26787978607034324,
                -0.30838768287984186,
                0.00989695818447031,
                -0.3411515688379077,
                0.08545800935661334,
                -0.38892505034367336,
            ],
            [
                -0.40579899720523244,
                0.02312562316595211,
                -0.5312022907527016,
                -0.3693065415065509,
                0.28778352515924405,
                -0.21408476966018827,
                0.08476590823896107,
                -0.331850261752073,
            ],
        ],
        [
            [
                -0.07732081532070446,
                0.29711704205119077,
                -0.0019518920644305531,
                -0.508875989551415,
                0.055048667001486175,
                -0.3703987511750884,
                0.08055090371865832,
                -0.325207265349378,
            ],
            [
                -0.30498842239321955,
                0.35928847744894904,
                -0.5660679969868672,
                -0.22544992794847946,
                0.4196988258521016,
                -0.2874247771733212,
                -0.11196088638313648,
                -0.3880208965645975,
            ],
            [
                -0.2141493311252869,
                -0.09908455318304277,
                -0.5954582803292919,
                -0.537354492372074,
                0.3239273501904272,
                -0.09249704773496159,
                0.11506418946857457,
                -0.3170834149460797,
            ],
        ],
    ]

    res_final_test = [
        [
            -0.07732081532070446,
            0.29711704205119077,
            -0.0019518920644305531,
            -0.508875989551415,
            0.055048667001486175,
            -0.3703987511750884,
            0.08055090371865832,
            -0.325207265349378,
        ],
        [
            -0.30498842239321955,
            0.35928847744894904,
            -0.5660679969868672,
            -0.22544992794847946,
            0.4196988258521016,
            -0.2874247771733212,
            -0.11196088638313648,
            -0.3880208965645975,
        ],
        [
            -0.2141493311252869,
            -0.09908455318304277,
            -0.5954582803292919,
            -0.537354492372074,
            0.3239273501904272,
            -0.09249704773496159,
            0.11506418946857457,
            -0.3170834149460797,
        ],
    ]

    res = [res_outputs, res_final_test]
    obj.run(res, data=inputs, cell=cell, time_major=True)
