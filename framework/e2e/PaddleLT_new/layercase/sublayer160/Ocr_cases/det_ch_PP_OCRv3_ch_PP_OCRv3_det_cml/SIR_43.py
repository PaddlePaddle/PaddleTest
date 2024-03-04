# api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.common.upsample||method:__add__||api:paddle.nn.functional.common.upsample||method:__add__||api:paddle.nn.functional.common.upsample||method:__add__||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||method:__add__||api:paddle.nn.functional.conv._conv_nd||method:__add__||api:paddle.nn.functional.conv._conv_nd||method:__add__||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.common.upsample||api:paddle.nn.functional.common.upsample||api:paddle.nn.functional.common.upsample||api:paddle.tensor.manipulation.concat
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[64, 64, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[64, 64, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[256, 2048, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[64, 64, 9, 9],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[64, 64, 9, 9],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[64, 256, 9, 9],
           dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
           shape=[64, 256, 9, 9],
           dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
           shape=[64, 256, 9, 9],
           dtype=paddle.float32,
        )
        self.parameter_8 = self.create_parameter(
           shape=[256, 256, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_9 = self.create_parameter(
           shape=[256, 512, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_10 = self.create_parameter(
           shape=[64, 256, 9, 9],
           dtype=paddle.float32,
        )
        self.parameter_11 = self.create_parameter(
           shape=[64, 64, 9, 9],
           dtype=paddle.float32,
        )
        self.parameter_12 = self.create_parameter(
           shape=[64, 64, 9, 9],
           dtype=paddle.float32,
        )
        self.parameter_13 = self.create_parameter(
           shape=[64, 64, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_14 = self.create_parameter(
           shape=[256, 1024, 1, 1],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [2, 256, 240, 240], dtype: paddle.float32, stop_gradient: True)
        var_1,    # (shape: [2, 512, 120, 120], dtype: paddle.float32, stop_gradient: True)
        var_2,    # (shape: [2, 1024, 60, 60], dtype: paddle.float32, stop_gradient: True)
        var_3,    # (shape: [2, 2048, 30, 30], dtype: paddle.float32, stop_gradient: True)
    ):
        var_4 = paddle.nn.functional.conv._conv_nd(var_3, self.parameter_2, bias=None, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_5 = paddle.nn.functional.conv._conv_nd(var_2, self.parameter_14, bias=None, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_6 = paddle.nn.functional.conv._conv_nd(var_1, self.parameter_9, bias=None, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_7 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_8, bias=None, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_8 = paddle.nn.functional.common.upsample(var_4, scale_factor=2, mode='nearest', align_mode=1)
        var_9 = var_5.__add__(var_8)
        var_10 = paddle.nn.functional.common.upsample(var_9, scale_factor=2, mode='nearest', align_mode=1)
        var_11 = var_6.__add__(var_10)
        var_12 = paddle.nn.functional.common.upsample(var_11, scale_factor=2, mode='nearest', align_mode=1)
        var_13 = var_7.__add__(var_12)
        var_14 = paddle.nn.functional.conv._conv_nd(var_4, self.parameter_6, bias=None, stride=[1, 1], padding=[4, 4], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_15 = paddle.nn.functional.conv._conv_nd(var_9, self.parameter_5, bias=None, stride=[1, 1], padding=[4, 4], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_16 = paddle.nn.functional.conv._conv_nd(var_11, self.parameter_7, bias=None, stride=[1, 1], padding=[4, 4], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_17 = paddle.nn.functional.conv._conv_nd(var_13, self.parameter_10, bias=None, stride=[1, 1], padding=[4, 4], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_18 = paddle.nn.functional.conv._conv_nd(var_17, self.parameter_1, bias=None, stride=[2, 2], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_19 = var_16.__add__(var_18)
        var_20 = paddle.nn.functional.conv._conv_nd(var_19, self.parameter_13, bias=None, stride=[2, 2], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_21 = var_15.__add__(var_20)
        var_22 = paddle.nn.functional.conv._conv_nd(var_21, self.parameter_0, bias=None, stride=[2, 2], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_23 = var_14.__add__(var_22)
        var_24 = paddle.nn.functional.conv._conv_nd(var_17, self.parameter_3, bias=None, stride=[1, 1], padding=[4, 4], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_25 = paddle.nn.functional.conv._conv_nd(var_19, self.parameter_11, bias=None, stride=[1, 1], padding=[4, 4], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_26 = paddle.nn.functional.conv._conv_nd(var_21, self.parameter_12, bias=None, stride=[1, 1], padding=[4, 4], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_27 = paddle.nn.functional.conv._conv_nd(var_23, self.parameter_4, bias=None, stride=[1, 1], padding=[4, 4], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_28 = paddle.nn.functional.common.upsample(var_27, scale_factor=8, mode='nearest', align_mode=1)
        var_29 = paddle.nn.functional.common.upsample(var_26, scale_factor=4, mode='nearest', align_mode=1)
        var_30 = paddle.nn.functional.common.upsample(var_25, scale_factor=2, mode='nearest', align_mode=1)
        var_31 = paddle.tensor.manipulation.concat([var_28, var_29, var_30, var_24], axis=1)
        return var_31


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[2, 256, 240, 240], dtype=paddle.float32),
        paddle.rand(shape=[2, 512, 120, 120], dtype=paddle.float32),
        paddle.rand(shape=[2, 1024, 60, 60], dtype=paddle.float32),
        paddle.rand(shape=[2, 2048, 30, 30], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[2, 256, 240, 240]).astype('float32'),
        np.random.random(size=[2, 512, 120, 120]).astype('float32'),
        np.random.random(size=[2, 1024, 60, 60]).astype('float32'),
        np.random.random(size=[2, 2048, 30, 30]).astype('float32'),
    )
    return inputs


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = create_paddle_inputs()
        self.net = LayerCase()
    def train(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.set_flags({'FLAGS_prim_all': with_prim})
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        paddle.seed(123)
        outs = net(*self.inputs)
        return outs
    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(self.net, to_static=True, with_prim=True, with_cinn=True)
        for st, cinn in zip(paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()