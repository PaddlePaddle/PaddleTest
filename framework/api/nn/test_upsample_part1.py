#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.nn.Upsample
"""
import numpy as np
import pytest
import paddle
from apibase import APIBase
from apibase import randtool


class TestUpsample(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.debug = True
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


obj = TestUpsample(paddle.nn.Upsample)


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


# @pytest.mark.api_nn_Upsample_vartype
# def test_upsample_linear_base():
#     """
#      mode = 'linear'
#
#      test all data types in self.types = [np.float32, np.float64]
#      which is defined in TestUpsample.hook().
#
#      data_format = 'NCW',
#      when size is None and scale_factor is not None,
#      test all combinations of align_corners and align_mode;
#      when size is not None and scale_factor is None,
#      test all combinations of align_corners and align_mode;
#     """
#     x = randtool("float", -10, 10, [2, 4, 10])
#     mode = "linear"
#     for data_format in ("NCW",):
#         for align_corners in (True, False):
#             for align_mode in (0, 1):
#                 # size is None and scale_factor is not None
#                 size = None
#                 for scale_factor in np.arange(0.6, 9, 0.1):
#                     scale_factor = [scale_factor.item()]
#                     res = linear_interpolation_using_numpy(
#                         x=x,
#                         size=size,
#                         scale_factor=scale_factor,
#                         align_corners=align_corners,
#                         align_mode=align_mode,
#                         data_format=data_format,
#                     )
#
#                     obj.base(
#                         res=res,
#                         data=x,
#                         size=size,
#                         scale_factor=scale_factor,
#                         mode=mode,
#                         align_corners=align_corners,
#                         align_mode=align_mode,
#                         data_format=data_format,
#                     )
#                 # size is not None and scale_factor is None
#                 scale_factor = None
#                 for size in range(2, 30, 3):
#                     size = [size]
#                     res = linear_interpolation_using_numpy(
#                         x=x,
#                         size=size,
#                         scale_factor=scale_factor,
#                         align_corners=align_corners,
#                         align_mode=align_mode,
#                         data_format=data_format,
#                     )
#                     obj.base(
#                         res=res,
#                         data=x,
#                         size=size,
#                         scale_factor=scale_factor,
#                         mode=mode,
#                         align_corners=align_corners,
#                         align_mode=align_mode,
#                         data_format=data_format,
#                     )


@pytest.mark.api_nn_Upsample_parameters
def test_upsample_linear():
    """
     mode = 'linear'

     data_format = 'NWC',
     when size is None and scale_factor is not None,
     test all combinations of align_corners and align_mode;
     when size is not None and scale_factor is None,
     test all combinations of align_corners and align_mode;
    """
    x = randtool("float", -10, 10, [2, 4, 10]).transpose((0, 2, 1))
    mode = "linear"
    data_format = "NWC"
    for align_corners in (True, False):
        for align_mode in (0, 1):
            # size is None and scale_factor is not None
            size = None
            for scale_factor in np.arange(0.4, 9, 0.2):
                scale_factor = [scale_factor.item()]
                res = linear_interpolation_using_numpy(
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
            for size in range(2, 30, 3):
                size = [size]
                res = linear_interpolation_using_numpy(
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
def test_upsample_linear2():
    """
    input has larger value range [-1000.0, 1000.0)
    """
    x = randtool("float", -1000, 1000, [2, 10, 4])
    size = [17]
    scale_factor = None
    mode = "linear"
    align_corners = True
    align_mode = 0
    data_format = "NWC"

    res = linear_interpolation_using_numpy(
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
def test_upsample_linear3():
    """
    size or scale_factor is a Tensor
    """
    x = randtool("float", -10, 10, [2, 10, 4])
    mode = "linear"
    align_corners = True
    align_mode = 0
    data_format = "NWC"

    for size, scale_factor in [([17], None), (None, [2.4])]:
        res = linear_interpolation_using_numpy(
            x=x,
            size=size,
            scale_factor=scale_factor,
            align_corners=align_corners,
            align_mode=align_mode,
            data_format=data_format,
        )
        # don't use dtype=np.int
        size = np.array(size, dtype=np.int32) if size is not None else None
        scale_factor = np.array(scale_factor, dtype=np.float32) if scale_factor is not None else None
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
def test_upsample_linear4():
    """
    ValueError: Only one of size or scale_factor should be defined.
    """
    x = randtool("float", -10, 10, [2, 4, 10])
    size = [15]
    scale_factor = [3.0]
    mode = "linear"
    align_corners = True
    align_mode = 0
    data_format = "NCW"
    kwargs = dict(
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        align_mode=align_mode,
        data_format=data_format,
    )
    obj.exception(etype=ValueError, mode_="python", data=x, **kwargs)


@pytest.mark.api_nn_Upsample_exception
def test_upsample_linear5():
    """
    ValueError: Got wrong value for param `data_format`: NCL received but only `NCW` or `NWC` supported for 3-D input.
    """
    x = randtool("float", -10, 10, [2, 4, 10])
    size = [15]
    scale_factor = None
    mode = "linear"
    align_corners = True
    align_mode = 0
    data_format = "NCL"
    kwargs = dict(
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        align_mode=align_mode,
        data_format=data_format,
    )
    obj.exception(etype=ValueError, mode_="python", data=x, **kwargs)


@pytest.mark.api_nn_Upsample_exception
def test_upsample_linear6():
    """
    ValueError: align_mode can only be 0 or 1
    """
    x = randtool("float", -10, 10, [2, 4, 10])
    size = [15]
    scale_factor = None
    mode = "linear"
    align_corners = True
    align_mode = 2
    data_format = "NCW"
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
def test_upsample_bilinear_base():
    """
     mode = 'bilinear'
     test all data types in self.types = [np.float32, np.float64]
     which is defined in TestUpsample.hook().

     data_format = 'NCHW',
     when size is None and scale_factor is not None,
     test all combinations of align_corners and align_mode;
     when size is not None and scale_factor is None,
     test all combinations of align_corners and align_mode;
    """
    x = randtool("float", -10, 10, [2, 2, 10, 10])
    mode = "bilinear"
    data_format = "NCHW"
    for align_corners in (True, False):
        for align_mode in (0, 1):
            # size is None and scale_factor is not None
            size = None
            for scale_h in np.arange(0.6, 4, 1.2):
                for scale_w in np.arange(0.6, 6, 1.2):
                    scale_factor = [scale_h.item(), scale_w.item()]
                    res = bilinear_interpolation_using_numpy(
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
            for out_h in range(2, 30, 11):
                for out_w in range(2, 30, 11):
                    size = [out_h, out_w]
                    res = bilinear_interpolation_using_numpy(
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
def test_upsample_bilinear():
    """
     mode = 'bilinear'

     data_format = 'NHWC',
     when size is None and scale_factor is not None,
     test all combinations of align_corners and align_mode;
     when size is not None and scale_factor is None,
     test all combinations of align_corners and align_mode;
    """
    x = randtool("float", -10, 10, [2, 2, 10, 10]).transpose((0, 2, 3, 1))
    mode = "bilinear"
    data_format = "NHWC"
    for align_corners in (True, False):
        for align_mode in (0, 1):
            # size is None and scale_factor is not None
            size = None
            for scale_h in np.arange(0.6, 4, 1.1):
                for scale_w in np.arange(0.6, 4, 1.1):
                    scale_factor = [scale_h.item(), scale_w.item()]
                    res = bilinear_interpolation_using_numpy(
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
            for out_h in range(2, 30, 12):
                for out_w in range(2, 30, 10):
                    size = [out_h, out_w]
                    res = bilinear_interpolation_using_numpy(
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
def test_upsample_bilinear2():
    """
    input has larger value range [-1000.0, 1000.0)
    """
    x = randtool("float", -1000, 1000, [2, 2, 4, 5])
    size = [10, 11]
    scale_factor = None
    mode = "bilinear"
    align_corners = True
    align_mode = 0
    data_format = "NCHW"

    res = bilinear_interpolation_using_numpy(
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
