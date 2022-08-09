#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.nn.Upsample
"""
import paddle
from apibase import randtool
from upsample_utils import *


obj = TestUpsample(paddle.nn.Upsample)


@pytest.mark.api_nn_Upsample_vartype
def test_upsample_linear_base():
    """
     mode = 'linear'

     test all data types in self.types = [np.float32, np.float64]
     which is defined in TestUpsample.hook().

     data_format = 'NCW',
     when size is None and scale_factor is not None,
     test all combinations of align_corners and align_mode;
     when size is not None and scale_factor is None,
     test all combinations of align_corners and align_mode;
    """
    x = randtool("float", -10, 10, [2, 4, 10]).transpose((0, 2, 1))
    mode = "linear"
    for data_format in ("NWC",):
        for align_corners in (True, False):
            for align_mode in (0, 1):
                # size is None and scale_factor is not None
                size = None
                for scale_factor in np.arange(0.6, 9, 0.1):
                    scale_factor = [scale_factor.item()]
                    res = linear_interpolation_using_numpy(
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
