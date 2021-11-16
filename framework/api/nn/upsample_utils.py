#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.nn.Upsample
"""
import numpy as np
import pytest
from apibase import APIBase


class TestUpsample(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.debug = False
        self.delta = 1e-3
        self.no_grad_var = ["scale_factor", "size"]

    def exception(self, etype, mode_="c", data=None, **kwargs):
        """
        overwrite this method to avoid argument name collision:
        change "mode" to "mode_",
        note: paddle.nn.Upsample also has an argument "mode"
        """
        # 禁止输入res
        if "res" in kwargs.keys():
            assert False, "exception检测不需要输入res参数"
        # 复用前向计算函数， 随便定义res
        res = np.array([0])
        if mode_ == "c":
            try:
                self.run(res, data, **kwargs)
            except Exception as e:
                e = str(e)
                if etype in e:
                    assert True
                else:
                    assert False, "异常校验失败,异常类型为" + etype
                # print(str(e))
        if mode_ == "python":
            with pytest.raises(etype):
                self.run(res, data, **kwargs)
                # print(excinfo.value)
                # assert "!23" in excinfo.value


def linear_interpolation_using_numpy(x, size, scale_factor=None, align_corners=True, align_mode=0, data_format="NCW"):
    """
    linear interpolation using numpy.
    x is a 3-D numpy.ndarray,
    size is a list/tuple of an integer,
    scale_factor is a list/tuple of a real number,
    there must be one and only one None in set(size, scale_factor).
    """
    if (size is None and scale_factor is None) or (size is not None and scale_factor is not None):
        assert False, "Only one of size or scale_factor should be defined"

    if data_format == "NWC":
        x = np.transpose(x, (0, 2, 1))  # NWC => NCW
    batch_size, channel, in_w = x.shape

    if size is not None:
        out_w = size[0]
        scale_w = 0
    else:
        scale_w = scale_factor[0]
        # Caution!!!!!
        # it's critical to add np.around, because C++ souce code of paddle.nn.Upsample
        # is like this:
        #      out_h = static_cast<int>(in_h * scale_h);
        #      out_w = static_cast<int>(in_w * scale_w);
        # so it's possible to have numerical instability to make the test fail
        out_w = np.int(np.around(in_w * scale_w))
    out = np.zeros((batch_size, channel, out_w))

    ratio_w = 0.0
    if out_w > 1:
        if align_corners:
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            if scale_w > 0:
                ratio_w = 1.0 / scale_w
            else:
                ratio_w = 1.0 * in_w / out_w

    for j in range(out_w):
        if align_mode == 0 and not align_corners:
            w = int(ratio_w * (j + 0.5) - 0.5)
        else:
            w = int(ratio_w * j)
        w = max(0, w)
        wid = 1 if w < in_w - 1 else 0

        if align_mode == 0 and not align_corners:
            idx_src_w = max(ratio_w * (j + 0.5) - 0.5, 0)
            w1lambda = idx_src_w - w
        else:
            w1lambda = ratio_w * j - w
        w2lambda = 1.0 - w1lambda

        out[:, :, j] = w2lambda * x[:, :, w] + w1lambda * x[:, :, w + wid]

    if data_format == "NWC":
        out = np.transpose(out, (0, 2, 1))  # NCW => NWC
    return out.astype(x.dtype)


def bilinear_interpolation_using_numpy(
    x, size, scale_factor=None, align_corners=True, align_mode=0, data_format="NCHW"
):
    """
    bilinear interpolation using numpy.
    x is a 4-D numpy.ndarray,
    size is a list/tuple of 2 integers,
    scale_factor is a list/tuple of 2 real number,
    there must be one and only one None in set(size, scale_factor).
    """
    if (size is None and scale_factor is None) or (size is not None and scale_factor is not None):
        assert False, "Only one of size or scale_factor should be defined"

    if data_format == "NHWC":
        x = np.transpose(x, (0, 3, 1, 2))  # NHWC => NCHW
    batch_size, channel, in_h, in_w = x.shape

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

    out = np.zeros((batch_size, channel, out_h, out_w))

    for i in range(out_h):
        if align_mode == 0 and not align_corners:
            h = int(ratio_h * (i + 0.5) - 0.5)
        else:
            h = int(ratio_h * i)
        h = max(0, h)
        hid = 1 if h < in_h - 1 else 0
        if align_mode == 0 and not align_corners:
            idx_src_h = max(ratio_h * (i + 0.5) - 0.5, 0)
            h1lambda = idx_src_h - h
        else:
            h1lambda = ratio_h * i - h
        h2lambda = 1.0 - h1lambda
        for j in range(out_w):
            if align_mode == 0 and not align_corners:
                w = int(ratio_w * (j + 0.5) - 0.5)
            else:
                w = int(ratio_w * j)
            w = max(0, w)
            wid = 1 if w < in_w - 1 else 0
            if align_mode == 0 and not align_corners:
                idx_src_w = max(ratio_w * (j + 0.5) - 0.5, 0)
                w1lambda = idx_src_w - w
            else:
                w1lambda = ratio_w * j - w
            w2lambda = 1.0 - w1lambda

            out[:, :, i, j] = h2lambda * (w2lambda * x[:, :, h, w] + w1lambda * x[:, :, h, w + wid]) + h1lambda * (
                w2lambda * x[:, :, h + hid, w] + w1lambda * x[:, :, h + hid, w + wid]
            )

    if data_format == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC
    return out.astype(x.dtype)


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
