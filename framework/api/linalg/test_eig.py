#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_eig
"""

import logging
import paddle
import numpy as np
import pytest


place = paddle.CPUPlace()


def cal_eig(dtype, x, place):
    """
    calculate eig api
    """
    paddle.device.set_device("cpu")
    x = x.astype(dtype)
    input = paddle.to_tensor(x, stop_gradient=False)
    dynamic_res = paddle.linalg.eig(input)
    dx = paddle.grad(dynamic_res, input)

    paddle.enable_static()
    main_program, startup_program = paddle.static.Program(), paddle.static.Program()
    with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
        x_s = paddle.static.data(name="x_s", shape=x.shape, dtype=dtype)
        y = paddle.linalg.eig(x_s)
        exe = paddle.static.Executor(place)
        exe.run(startup_program)
        static_res = exe.run(main_program, feed={"x_s": x}, fetch_list=[y], return_numpy=True)
        paddle.disable_static()
        length = len(dynamic_res)
        for i in range(length):
            assert np.allclose(dynamic_res[i].numpy(), static_res[i])
    # logging.info(dynamic_res)
    return static_res, dx


@pytest.mark.api_linalg_eig_vartype
def test_eig_base():
    """
    base
    """
    types = ["float32", "float64", "complex64", "complex128"]
    np.random.seed(22)
    A = np.random.rand(4, 4) * 10
    res = np.linalg.eig(A)
    for d in types:
        (e_value, e_vector), dx = cal_eig(d, A, place)
        assert np.allclose(res[0], e_value)
        assert np.allclose(np.abs(res[1]), np.abs(e_vector))
        if d == "float64":
            assert np.allclose(
                dx,
                np.array(
                    [
                        [0.94879350, -0.02079626, 0.14372541, -0.02199845],
                        [-0.06709838, 1.07928009, 0.06943255, -0.05609993],
                        [-0.01982539, -0.05201412, 1.00906748, 0.07527123],
                        [-0.06332734, 0.02413055, 0.11526301, 0.96285893],
                    ]
                ),
            )


@pytest.mark.api_linalg_eig_vartype
def test_eig0():
    """
    default
    """

    np.random.seed(122)
    A = np.random.rand(2, 2)
    res = np.linalg.eig(A)
    (e_value, e_vector), dx = cal_eig("float64", A, place)
    assert np.allclose(res[0], e_value)
    assert np.allclose(np.abs(res[1]), np.abs(e_vector))
    assert np.allclose(dx, np.array([[1.59156463, -0.18015260], [1.15057946, 0.40843537]]))


@pytest.mark.api_linalg_eig_vartype
def test_eig1():
    """
    complex128
    """

    np.random.seed(2)
    A = np.random.rand(3, 3)
    A = 4 * A + A * 1j
    res = np.linalg.eig(A)
    (e_value, e_vector), dx = cal_eig("complex128", A, place)
    assert np.allclose(res[0], e_value)
    assert np.allclose(np.abs(res[1]), np.abs(e_vector))
    assert np.allclose(
        dx,
        np.array(
            [
                [
                    (0.7439317178184834 - 0.06401707054537936j),
                    (0.03053219960527948 + 0.007633049901320199j),
                    (0.24658921340314144 + 0.06164730335078566j),
                ],
                [
                    (-0.31156720547878697 - 0.07789180136969652j),
                    (1.0096094440510113 + 0.0024023610127527073j),
                    (0.2409878720351644 + 0.060246968008791105j),
                ],
                [
                    (-0.29670688087238234 - 0.07417672021809588j),
                    (0.022875531919249768 + 0.005718882979812565j),
                    (1.2464588381305055 + 0.06161470953262671j),
                ],
            ]
        ),
    )


@pytest.mark.api_linalg_eig_vartype
def test_eig2():
    """
    multiple dimension
    complex128
    """

    np.random.seed(2)
    A = np.random.rand(4, 3, 3)
    A = 3 * A + A * 4j
    res = np.linalg.eig(A)
    (e_value, e_vector), dx = cal_eig("complex128", A, place)
    assert np.allclose(res[0], e_value)
    assert np.allclose(np.abs(res[1]), np.abs(e_vector))
    assert np.allclose(
        dx,
        np.array(
            [
                [
                    [
                        (0.8694051760874267 - 0.17412643188343122j),
                        (0.015571421798692444 + 0.02076189573159036j),
                        (0.12576049883560209 + 0.16768066511413637j),
                    ],
                    [
                        (-0.15889927479418173 - 0.21186569972557528j),
                        (1.0049008164660154 + 0.006534421954687619j),
                        (0.12290381473793394 + 0.16387175298391177j),
                    ],
                    [
                        (-0.15132050924491475 - 0.2017606789932199j),
                        (0.011666521278817212 + 0.015555361705089744j),
                        (1.1256940074465576 + 0.16759200992874368j),
                    ],
                ],
                [
                    [
                        (1.0470881636474823 + 0.06278421819664316j),
                        (0.040096644727220083 + 0.05346219296962667j),
                        (-0.11027939185334978 - 0.14703918913779934j),
                    ],
                    [
                        (-0.042255824208871626 - 0.05634109894516216j),
                        (1.1007869947429334 + 0.13438265965724444j),
                        (-0.027748911803145124 - 0.03699854907085994j),
                    ],
                    [
                        (0.006449175389578704 + 0.008598900519438237j),
                        (0.05614572567789428 + 0.07486096757052584j),
                        (0.8521248416095842 - 0.19716687785388767j),
                    ],
                ],
                [
                    [
                        (0.8331497649358067 - 0.22246698008559124j),
                        (0.0929561009610675 + 0.12394146794809006j),
                        (-0.0032766351330119205 - 0.004368846844015935j),
                    ],
                    [
                        (0.11076826975395203 + 0.14769102633860284j),
                        (1.143440813200562 + 0.19125441760074957j),
                        (0.09530346046808115 + 0.127071280624109j),
                    ],
                    [
                        (0.014089101665084128 + 0.018785468886778588j),
                        (0.08475468154309834 + 0.11300624205746469j),
                        (1.0234094218636312 + 0.03121256248484156j),
                    ],
                ],
                [
                    [
                        (0.9131949493806791 - 0.1157400674924277j),
                        (-0.009514918809063815 - 0.012686558412084804j),
                        (0.12770508149432216 + 0.17027344199242916j),
                    ],
                    [
                        (0.04834868689869894 + 0.06446491586493137j),
                        (0.9217900187513551 - 0.10427997499819304j),
                        (0.006209995694866392 + 0.008279994259821927j),
                    ],
                    [
                        (-0.24306992880265066 - 0.32409323840353393j),
                        (0.018736117730800392 + 0.024981490307733868j),
                        (1.1650150318679655 + 0.22002004249062063j),
                    ],
                ],
            ]
        ),
    )
