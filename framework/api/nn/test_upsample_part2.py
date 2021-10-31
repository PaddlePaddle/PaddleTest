#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.nn.Upsample
"""
import numpy as np
import pytest
import paddle
from apibase import randtool
from test_upsample_part1 import TestUpsample


obj = TestUpsample(paddle.nn.Upsample)


def trilinear_interpolation_using_numpy(
    x, size, scale_factor=None, align_corners=True, align_mode=0, data_format="NCDHW"
):
    """
    trilinear interpolation using numpy.
    x is a 5-D numpy.ndarray,
    size is a list/tuple of 3 integers,
    scale_factor is a list/tuple of 3 real numbers,
    there must be one and only one None in set(size, scale_factor).
    """
    if (size is None and scale_factor is None) or (size is not None and scale_factor is not None):
        assert False, "Only one of size or scale_factor should be defined"

    if data_format == "NDHWC":
        x = np.transpose(x, (0, 4, 1, 2, 3))  # NDHWC => NCDHW
    batch_size, channel, in_d, in_h, in_w = x.shape

    if size is not None:
        out_d, out_h, out_w = size
        scale_d, scale_h, scale_w = 0, 0, 0
    else:
        scale_d, scale_h, scale_w = scale_factor
        # Caution!!!!!
        # it's critical to add np.around, because C++ souce code of paddle.nn.Upsample
        # is like this:
        #      out_h = static_cast<int>(in_h * scale_h);
        #      out_w = static_cast<int>(in_w * scale_w);
        # so it's possible to have numerical instability to make the test fail
        out_d = np.int(np.around(in_d * scale_d))
        out_h = np.int(np.around(in_h * scale_h))
        out_w = np.int(np.around(in_w * scale_w))

    ratio_d = ratio_h = ratio_w = 0.0
    if out_d > 1:
        if align_corners:
            ratio_d = (in_d - 1.0) / (out_d - 1.0)
        else:
            if scale_d > 0:
                ratio_d = 1.0 / scale_d
            else:
                ratio_d = 1.0 * in_d / out_d
    if out_h > 1:
        if align_corners:
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        else:
            if scale_h > 0:
                ratio_h = 1.0 / scale_h
            else:
                ratio_h = 1.0 * in_h / out_h
    if out_w > 1:
        if align_corners:
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            if scale_w > 0:
                ratio_w = 1.0 / scale_w
            else:
                ratio_w = 1.0 * in_w / out_w

    out = np.zeros((batch_size, channel, out_d, out_h, out_w))

    for i in range(out_d):
        if align_mode == 0 and not align_corners:
            d = int(ratio_d * (i + 0.5) - 0.5)
        else:
            d = int(ratio_d * i)

        d = max(0, d)
        did = 1 if d < in_d - 1 else 0
        if align_mode == 0 and not align_corners:
            idx_src_d = max(ratio_d * (i + 0.5) - 0.5, 0)
            d1lambda = idx_src_d - d
        else:
            d1lambda = ratio_d * i - d
        d2lambda = 1.0 - d1lambda

        for j in range(out_h):
            if align_mode == 0 and not align_corners:
                h = int(ratio_h * (j + 0.5) - 0.5)
            else:
                h = int(ratio_h * j)

            h = max(0, h)
            hid = 1 if h < in_h - 1 else 0
            if align_mode == 0 and not align_corners:
                idx_src_h = max(ratio_h * (j + 0.5) - 0.5, 0)
                h1lambda = idx_src_h - h
            else:
                h1lambda = ratio_h * j - h
            h2lambda = 1.0 - h1lambda

            for k in range(out_w):
                if align_mode == 0 and not align_corners:
                    w = int(ratio_w * (k + 0.5) - 0.5)
                else:
                    w = int(ratio_w * k)
                w = max(0, w)
                wid = 1 if w < in_w - 1 else 0
                if align_mode == 0 and not align_corners:
                    idx_src_w = max(ratio_w * (k + 0.5) - 0.5, 0)
                    w1lambda = idx_src_w - w
                else:
                    w1lambda = ratio_w * k - w
                w2lambda = 1.0 - w1lambda

                out[:, :, i, j, k] = d2lambda * (
                    h2lambda * (w2lambda * x[:, :, d, h, w] + w1lambda * x[:, :, d, h, w + wid])
                    + h1lambda * (w2lambda * x[:, :, d, h + hid, w] + w1lambda * x[:, :, d, h + hid, w + wid])
                ) + d1lambda * (
                    h2lambda * (w2lambda * x[:, :, d + did, h, w] + w1lambda * x[:, :, d + did, h, w + wid])
                    + h1lambda
                    * (w2lambda * x[:, :, d + did, h + hid, w] + w1lambda * x[:, :, d + did, h + hid, w + wid])
                )
    if data_format == "NDHWC":
        out = np.transpose(out, (0, 2, 3, 4, 1))  # NCDHW => NDHWC
    return out.astype(x.dtype)


def nearest_neighbor_interpolation_using_numpy(x, size, scale_factor=None, align_corners=True, data_format="NCHW"):
    """
    nearest neighbor interpolation using numpy.
    x is a 4-D numpy.ndarray,
    size is None or a list/tuple of 2 integers,
    scale_factor is None or a list/tuple of 2 real numbers ,
    there must be one and only one None in set(size, scale_factor).
    """
    if (size is None and scale_factor is None) or (size is not None and scale_factor is not None):
        assert False, "Only one of size or scale_factor should be defined"

    if data_format == "NHWC":
        x = np.transpose(x, (0, 3, 1, 2))  # NHWC => NCHW
    batch_size, channels, in_h, in_w = x.shape

    if size is not None:
        out_h, out_w = size
        scale_h, scale_w = 0, 0
    else:
        scale_h, scale_w = scale_factor
        # Caution!!!!!
        # it's critical to add np.around, because C++ souce code of paddle.nn.Upsample
        # is like this:
        #      out_h = static_cast<int>(in_h * scale_h);
        #      out_w = static_cast<int>(in_w * scale_w);
        # so it's possible to have numerical instability to make the test fail
        out_h = np.int(np.around(in_h * scale_h))
        out_w = np.int(np.around(in_w * scale_w))

    ratio_h = ratio_w = 0.0
    if out_h > 1:
        if align_corners:
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        else:
            if scale_h > 0:
                ratio_h = 1.0 / scale_h
            else:
                ratio_h = 1.0 * in_h / out_h
    if out_w > 1:
        if align_corners:
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            if scale_w > 0:
                ratio_w = 1.0 / scale_w
            else:
                ratio_w = 1.0 * in_w / out_w
    out = np.zeros((batch_size, channels, out_h, out_w))

    if align_corners:
        for i in range(out_h):
            in_i = int(ratio_h * i + 0.5)
            for j in range(out_w):
                in_j = int(ratio_w * j + 0.5)
                out[:, :, i, j] = x[:, :, in_i, in_j]
    else:
        for i in range(out_h):
            in_i = int(ratio_h * i)
            for j in range(out_w):
                in_j = int(ratio_w * j)
                out[:, :, i, j] = x[:, :, in_i, in_j]

    if data_format == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC
    return out.astype(x.dtype)


def cubic_1(x, a):
    """
    x, a: real number
    """
    return ((a + 2) * x - (a + 3)) * x * x + 1


def cubic_2(x, a):
    """
    x, a: real number
    """
    return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a


def cubic_interp1d(x0, x1, x2, x3, t):
    """
    x0, x1, x2, x3, t: real numbers
    """
    param = [0, 0, 0, 0]
    a = -0.75
    x_1 = t
    x_2 = 1.0 - t
    param[0] = cubic_2(x_1 + 1.0, a)
    param[1] = cubic_1(x_1, a)
    param[2] = cubic_1(x_2, a)
    param[3] = cubic_2(x_2 + 1.0, a)
    return x0 * param[0] + x1 * param[1] + x2 * param[2] + x3 * param[3]


def bicubic_interpolation_using_numpy(x, size, scale_factor=None, align_corners=True, data_format="NCHW"):
    """
    bicubic interpolation using numpy.
    x is a 4-D numpy.ndarray,
    size is None or a list/tuple of 2 integers,
    scale_factor is None or a list/tuple of 2 real numbers ,
    there must be one and only one None in set(size, scale_factor).
    """
    if (size is None and scale_factor is None) or (size is not None and scale_factor is not None):
        assert False, "Only one of size or scale_factor should be defined"

    if data_format == "NHWC":
        x = np.transpose(x, (0, 3, 1, 2))  # NHWC => NCHW
    batch_size, channels, in_h, in_w = x.shape

    if size is not None:
        out_h, out_w = size
        scale_h, scale_w = 0, 0
    else:
        scale_h, scale_w = scale_factor
        # Caution!!!!!
        # it's critical to add np.around, because C++ souce code of paddle.nn.Upsample
        # is like this:
        #      out_h = static_cast<int>(in_h * scale_h);
        #      out_w = static_cast<int>(in_w * scale_w);
        # so it's possible to have numerical instability to make the test fail
        out_h = np.int(np.around(in_h * scale_h))
        out_w = np.int(np.around(in_w * scale_w))

    ratio_h = ratio_w = 0.0
    if out_h > 1:
        if align_corners:
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        else:
            if scale_h > 0:
                ratio_h = 1.0 / scale_h
            else:
                ratio_h = 1.0 * in_h / out_h

    if out_w > 1:
        if align_corners:
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            if scale_w > 0:
                ratio_w = 1.0 / scale_w
            else:
                ratio_w = 1.0 * in_w / out_w

    out = np.zeros((batch_size, channels, out_h, out_w))

    for k in range(out_h):
        if align_corners:
            h = ratio_h * k
        else:
            h = ratio_h * (k + 0.5) - 0.5
        input_y = np.floor(h)
        y_t = h - input_y
        for p in range(out_w):
            if align_corners:
                w = ratio_w * p
            else:
                w = ratio_w * (p + 0.5) - 0.5
            input_x = np.floor(w)
            x_t = w - input_x
            for i in range(batch_size):
                for j in range(channels):
                    coefficients = [0, 0, 0, 0]
                    access_x_0 = int(max(min(input_x - 1, in_w - 1), 0))
                    access_x_1 = int(max(min(input_x + 0, in_w - 1), 0))
                    access_x_2 = int(max(min(input_x + 1, in_w - 1), 0))
                    access_x_3 = int(max(min(input_x + 2, in_w - 1), 0))
                    for ii in range(4):
                        access_y = int(max(min(input_y - 1 + ii, in_h - 1), 0))
                        coefficients[ii] = cubic_interp1d(
                            x[i, j, access_y, access_x_0],
                            x[i, j, access_y, access_x_1],
                            x[i, j, access_y, access_x_2],
                            x[i, j, access_y, access_x_3],
                            x_t,
                        )
                    out[i, j, k, p] = cubic_interp1d(
                        coefficients[0], coefficients[1], coefficients[2], coefficients[3], y_t
                    )
    if data_format == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC
    return out.astype(x.dtype)


@pytest.mark.api_nn_Upsample_vartype
def test_upsample_trilinear_base():
    """
     mode = 'trilinear'

     test all data types in self.types = [np.float32, np.float64]
     which is defined in TestUpsample.hook().

     data_format = 'NCDHW',
     when size is None and scale_factor is not None,
     test all combinations of align_corners and align_mode;
     when size is not None and scale_factor is None,
     test all combinations of align_corners and align_mode;
    """
    x = randtool("float", -10, 10, [2, 4, 10, 10, 10])
    mode = "trilinear"
    data_format = "NCDHW"
    for align_corners in (True, False):
        for align_mode in (0, 1):
            # size is None and scale_factor is not None
            size = None
            for scale_d in np.arange(0.6, 1, 1.2):
                for scale_h in np.arange(0.6, 1, 1):
                    for scale_w in np.arange(0.6, 1, 0.9):
                        scale_factor = [scale_d.item(), scale_h.item(), scale_w.item()]
                        res = trilinear_interpolation_using_numpy(
                            x=x,
                            size=size,
                            scale_factor=scale_factor,
                            align_corners=align_corners,
                            align_mode=align_mode,
                            data_format=data_format,
                        )

                        obj.base(
                            res=res,
                            data=x,
                            size=size,
                            scale_factor=scale_factor,
                            mode=mode,
                            align_corners=align_corners,
                            align_mode=align_mode,
                            data_format=data_format,
                        )

            # size is not None and scale_factor is None
            scale_factor = None
            for out_d in range(2, 10, 13):
                for out_h in range(2, 10, 13):
                    for out_w in range(2, 10, 13):
                        size = [out_d, out_h, out_w]
                        res = trilinear_interpolation_using_numpy(
                            x=x,
                            size=size,
                            scale_factor=scale_factor,
                            align_corners=align_corners,
                            align_mode=align_mode,
                            data_format=data_format,
                        )
                        obj.base(
                            res=res,
                            data=x,
                            size=size,
                            scale_factor=scale_factor,
                            mode=mode,
                            align_corners=align_corners,
                            align_mode=align_mode,
                            data_format=data_format,
                        )


@pytest.mark.api_nn_Upsample_parameters
def test_upsample_trilinear():
    """
     mode = 'trilinear'

     data_format = 'NDHWC',
     when size is None and scale_factor is not None,
     test all combinations of align_corners and align_mode;
     when size is not None and scale_factor is None,
     test all combinations of align_corners and align_mode;
    """
    x = randtool("float", -10, 10, [2, 4, 10, 10, 10]).transpose((0, 2, 3, 4, 1))
    mode = "trilinear"
    data_format = "NDHWC"
    for align_corners in (True, False):
        for align_mode in (0, 1):
            # size is None and scale_factor is not None
            size = None
            for scale_d in np.arange(0.6, 1, 1.2):
                for scale_h in np.arange(1.6, 2, 2):
                    for scale_w in np.arange(1.7, 2, 0.9):
                        scale_factor = [scale_d.item(), scale_h.item(), scale_w.item()]
                        res = trilinear_interpolation_using_numpy(
                            x=x,
                            size=size,
                            scale_factor=scale_factor,
                            align_corners=align_corners,
                            align_mode=align_mode,
                            data_format=data_format,
                        )

                        obj.run(
                            res=res,
                            data=x,
                            size=size,
                            scale_factor=scale_factor,
                            mode=mode,
                            align_corners=align_corners,
                            align_mode=align_mode,
                            data_format=data_format,
                        )

            # size is not None and scale_factor is None
            scale_factor = None
            for out_d in range(4, 9, 10):
                for out_h in range(2, 5, 5):
                    for out_w in range(3, 5, 5):
                        size = [out_d, out_h, out_w]
                        res = trilinear_interpolation_using_numpy(
                            x=x,
                            size=size,
                            scale_factor=scale_factor,
                            align_corners=align_corners,
                            align_mode=align_mode,
                            data_format=data_format,
                        )
                        obj.run(
                            res=res,
                            data=x,
                            size=size,
                            scale_factor=scale_factor,
                            mode=mode,
                            align_corners=align_corners,
                            align_mode=align_mode,
                            data_format=data_format,
                        )


@pytest.mark.api_nn_Upsample_parameters
def test_upsample_trilinear2():
    """
    input has larger value range [-1000.0, 1000.0)
    """
    x = randtool("float", -1000, 1000, [2, 2, 4, 5, 6])
    size = [9, 10, 11]
    scale_factor = None
    mode = "trilinear"
    align_corners = False
    align_mode = 0
    data_format = "NCDHW"

    res = trilinear_interpolation_using_numpy(
        x=x,
        size=size,
        scale_factor=scale_factor,
        align_corners=align_corners,
        align_mode=align_mode,
        data_format=data_format,
    )
    obj.run(
        res=res,
        data=x,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        align_mode=align_mode,
        data_format=data_format,
    )


@pytest.mark.api_nn_Upsample_vartype
def test_upsample_nearest_base():
    """
     mode = 'nearest'

     test all data types in self.types = [np.float32, np.float64]
     which is defined in TestUpsample.hook().

     data_format = 'NCHW',
     when size is None and scale_factor is not None,
     test all combinations of align_corners and align_mode;
     when size is not None and scale_factor is None,
     test all combinations of align_corners and align_mode;

     note:
     when using mode = 'nearest', align_corners must be set False;
     when using mode = 'nearest', align_mode has no effect
    """
    x = randtool("float", -10, 10, [2, 2, 10, 10])
    mode = "nearest"
    data_format = "NCHW"
    align_corners = False
    # size is None and scale_factor is not None
    size = None
    for scale_h in np.arange(0.6, 3, 1.2):
        for scale_w in np.arange(0.6, 3, 1.2):
            scale_factor = [scale_h.item(), scale_w.item()]
            res = nearest_neighbor_interpolation_using_numpy(
                x=x, size=size, scale_factor=scale_factor, align_corners=align_corners, data_format=data_format
            )
            for align_mode in (0, 1):
                obj.base(
                    res=res,
                    data=x,
                    size=size,
                    scale_factor=scale_factor,
                    mode=mode,
                    align_corners=align_corners,
                    align_mode=align_mode,
                    data_format=data_format,
                )
    # size is not None and scale_factor is None
    scale_factor = None
    for out_h in range(2, 20, 11):
        for out_w in range(2, 20, 11):
            size = [out_h, out_w]
            res = nearest_neighbor_interpolation_using_numpy(
                x=x, size=size, scale_factor=scale_factor, align_corners=align_corners, data_format=data_format
            )
            for align_mode in (0, 1):
                obj.base(
                    res=res,
                    data=x,
                    size=size,
                    scale_factor=scale_factor,
                    mode=mode,
                    align_corners=align_corners,
                    align_mode=align_mode,
                    data_format=data_format,
                )


@pytest.mark.api_nn_Upsample_parameters
def test_upsample_nearest():
    """
     mode = 'nearest'

     data_format = 'NHWC',

     when size is None and scale_factor is not None,
     test all combinations of align_corners and align_mode;
     when size is not None and scale_factor is None,
     test all combinations of align_corners and align_mode;

     note:
     when using mode = 'nearest', align_corners must be set False;
     when using mode = 'nearest', align_mode has no effect
    """
    x = randtool("float", -10, 10, [2, 2, 10, 10]).transpose((0, 2, 3, 1))
    mode = "NeAReSt"
    data_format = "NHWC"
    align_corners = False
    # size is None and scale_factor is not None
    size = None
    for scale_h in np.arange(0.6, 3, 1.2):
        for scale_w in np.arange(0.6, 3, 1.2):
            scale_factor = [scale_h.item(), scale_w.item()]
            res = nearest_neighbor_interpolation_using_numpy(
                x=x, size=size, scale_factor=scale_factor, align_corners=align_corners, data_format=data_format
            )
            for align_mode in (0, 1):
                obj.run(
                    res=res,
                    data=x,
                    size=size,
                    scale_factor=scale_factor,
                    mode=mode,
                    align_corners=align_corners,
                    align_mode=align_mode,
                    data_format=data_format,
                )
    # size is not None and scale_factor is None
    scale_factor = None
    for out_h in range(2, 20, 11):
        for out_w in range(2, 20, 11):
            size = [out_h, out_w]
            res = nearest_neighbor_interpolation_using_numpy(
                x=x, size=size, scale_factor=scale_factor, align_corners=align_corners, data_format=data_format
            )
            for align_mode in (0, 1):
                obj.run(
                    res=res,
                    data=x,
                    size=size,
                    scale_factor=scale_factor,
                    mode=mode,
                    align_corners=align_corners,
                    align_mode=align_mode,
                    data_format=data_format,
                )


@pytest.mark.api_nn_Upsample_parameters
def test_upsample_nearest2():
    """
    mode = 'nearest'
    input has larger value range [-1000.0, 1000.0)

    note:
    when using mode = 'nearest', align_corners must be set False;
    when using mode = 'nearest', align_mode has no effect
    """
    x = randtool("float", -1000, 1000, [2, 2, 4, 5])
    size = [9, 10]
    scale_factor = None
    mode = "nearest"
    align_corners = False
    align_mode = 0
    data_format = "NCHW"

    res = nearest_neighbor_interpolation_using_numpy(
        x=x, size=size, scale_factor=scale_factor, align_corners=align_corners, data_format=data_format
    )
    obj.run(
        res=res,
        data=x,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        align_mode=align_mode,
        data_format=data_format,
    )


@pytest.mark.api_nn_Upsample_exception
def test_upsample_nearest3():
    """
    mode = 'nearest'
    align_corners = True

    ValueError: align_corners option can only be set with the interpolating modes:
    linear | bilinear | bicubic | trilinear.

    note:
    when using mode = 'nearest', align_corners must be set False;
    when using mode = 'nearest', align_mode has no effect
    """
    x = randtool("float", -10, 10, [2, 4, 5, 7])
    size = [10, 4]
    scale_factor = None
    mode = "neAresT"
    align_corners = True
    align_mode = 1
    data_format = "NCHW"
    kwargs = dict(
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        align_mode=align_mode,
        data_format=data_format,
    )
    obj.exception(etype=ValueError, mode_="python", data=x, **kwargs)


@pytest.mark.api_nn_Upsample_vartype
def test_upsample_bicubic_base():
    """
     mode = 'bicubic'

     test all data types in self.types = [np.float32, np.float64]
     which is defined in TestUpsample.hook().

     data_format = 'NCHW',
     when size is None and scale_factor is not None,
     test all combinations of align_corners and align_mode;
     when size is not None and scale_factor is None,
     test all combinations of align_corners and align_mode;

    note:
    when using mode = 'bicubic', align_corners can be set False or True;
    when using mode = 'bicubic', align_mode has no effect
    """
    x = randtool("float", -10, 10, [2, 2, 10, 10])
    mode = "bicubic"
    data_format = "NCHW"
    for align_corners in (False, True):
        # size is None and scale_factor is not None
        size = None
        for scale_h in np.arange(0.6, 3, 1.2):
            for scale_w in np.arange(0.6, 3, 1.2):
                scale_factor = [scale_h.item(), scale_w.item()]
                res = bicubic_interpolation_using_numpy(
                    x=x, size=size, scale_factor=scale_factor, align_corners=align_corners, data_format=data_format
                )
                for align_mode in (0, 1):
                    obj.base(
                        res=res,
                        data=x,
                        size=size,
                        scale_factor=scale_factor,
                        mode=mode,
                        align_corners=align_corners,
                        align_mode=align_mode,
                        data_format=data_format,
                    )
        # size is not None and scale_factor is None
        scale_factor = None
        for out_h in range(2, 20, 11):
            for out_w in range(2, 20, 11):
                size = [out_h, out_w]
                res = bicubic_interpolation_using_numpy(
                    x=x, size=size, scale_factor=scale_factor, align_corners=align_corners, data_format=data_format
                )
                for align_mode in (0, 1):
                    obj.base(
                        res=res,
                        data=x,
                        size=size,
                        scale_factor=scale_factor,
                        mode=mode,
                        align_corners=align_corners,
                        align_mode=align_mode,
                        data_format=data_format,
                    )


@pytest.mark.api_nn_Upsample_parameters
def test_upsample_bicubic():
    """
     mode = 'bicubic'

     data_format = 'NHWC',
     when size is None and scale_factor is not None,
     test all combinations of align_corners and align_mode;
     when size is not None and scale_factor is None,
     test all combinations of align_corners and align_mode;

    note:
    when using mode = 'bicubic', align_corners can be set False or True;
    when using mode = 'bicubic', align_mode has no effect
    """
    x = randtool("float", -10, 10, [2, 2, 10, 10]).transpose((0, 2, 3, 1))
    mode = "bicubic"
    data_format = "NHWC"
    for align_corners in (False, True):
        # size is None and scale_factor is not None
        size = None
        for scale_h in np.arange(0.6, 3, 1.2):
            for scale_w in np.arange(0.6, 3, 1.2):
                scale_factor = [scale_h.item(), scale_w.item()]
                res = bicubic_interpolation_using_numpy(
                    x=x, size=size, scale_factor=scale_factor, align_corners=align_corners, data_format=data_format
                )
                for align_mode in (0, 1):
                    obj.run(
                        res=res,
                        data=x,
                        size=size,
                        scale_factor=scale_factor,
                        mode=mode,
                        align_corners=align_corners,
                        align_mode=align_mode,
                        data_format=data_format,
                    )
        # size is not None and scale_factor is None
        scale_factor = None
        for out_h in range(2, 20, 11):
            for out_w in range(2, 20, 11):
                size = [out_h, out_w]
                res = bicubic_interpolation_using_numpy(
                    x=x, size=size, scale_factor=scale_factor, align_corners=align_corners, data_format=data_format
                )
                for align_mode in (0, 1):
                    obj.run(
                        res=res,
                        data=x,
                        size=size,
                        scale_factor=scale_factor,
                        mode=mode,
                        align_corners=align_corners,
                        align_mode=align_mode,
                        data_format=data_format,
                    )


@pytest.mark.api_nn_Upsample_parameters
def test_upsample_bicubic2():
    """
    mode = 'bicubic'
    input has larger value range [-1000.0, 1000.0)

    note:
    when using mode = 'bicubic', align_corners can be set False or True;
    when using mode = 'bicubic', align_mode has no effect
    """
    x = randtool("float", -1000, 1000, [2, 2, 4, 5])
    size = [9, 10]
    scale_factor = None
    mode = "bicubic"
    align_corners = False
    align_mode = 0
    data_format = "NCHW"

    res = bicubic_interpolation_using_numpy(
        x=x, size=size, scale_factor=scale_factor, align_corners=align_corners, data_format=data_format
    )
    obj.run(
        res=res,
        data=x,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        align_mode=align_mode,
        data_format=data_format,
    )
