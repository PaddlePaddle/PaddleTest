# api:paddle.nn.functional.common.interpolate||api:paddle.nn.functional.common.interpolate||api:paddle.nn.functional.common.interpolate||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.pooling.avg_pool2d||api:paddle.nn.functional.pooling.avg_pool2d||api:paddle.nn.functional.pooling.avg_pool2d||api:paddle.nn.functional.pooling.avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[256, 270, 1, 1],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 18, 192, 288], dtype: paddle.float32, stop_gradient: True)
        var_1,    # (shape: [1, 36, 96, 144], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [1, 72, 48, 72], dtype: paddle.float32, stop_gradient: False)
        var_3,    # (shape: [1, 144, 24, 36], dtype: paddle.float32, stop_gradient: False)
    ):
        var_4 = paddle.nn.functional.common.interpolate(var_1, scale_factor=2, mode='bilinear')
        var_5 = paddle.nn.functional.common.interpolate(var_2, scale_factor=4, mode='bilinear')
        var_6 = paddle.nn.functional.common.interpolate(var_3, scale_factor=8, mode='bilinear')
        var_7 = paddle.tensor.manipulation.concat([var_0, var_4, var_5, var_6], axis=1)
        var_8 = paddle.nn.functional.conv._conv_nd(var_7, self.parameter_5, bias=None, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_9 = paddle.nn.functional.pooling.avg_pool2d(var_8, kernel_size=2, stride=2)
        var_10 = paddle.nn.functional.pooling.avg_pool2d(var_8, kernel_size=4, stride=4)
        var_11 = paddle.nn.functional.pooling.avg_pool2d(var_8, kernel_size=8, stride=8)
        var_12 = paddle.nn.functional.pooling.avg_pool2d(var_8, kernel_size=16, stride=16)
        var_13 = paddle.nn.functional.conv._conv_nd(var_8, self.parameter_0, bias=None, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_14 = paddle.nn.functional.conv._conv_nd(var_9, self.parameter_3, bias=None, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_15 = paddle.nn.functional.conv._conv_nd(var_10, self.parameter_2, bias=None, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_16 = paddle.nn.functional.conv._conv_nd(var_11, self.parameter_1, bias=None, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_17 = paddle.nn.functional.conv._conv_nd(var_12, self.parameter_4, bias=None, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        return var_13, var_14, var_15, var_16, var_17


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 18, 192, 288], dtype=paddle.float32),
        paddle.rand(shape=[1, 36, 96, 144], dtype=paddle.float32),
        paddle.rand(shape=[1, 72, 48, 72], dtype=paddle.float32),
        paddle.rand(shape=[1, 144, 24, 36], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 18, 192, 288]).astype('float32'),
        np.random.random(size=[1, 36, 96, 144]).astype('float32'),
        np.random.random(size=[1, 72, 48, 72]).astype('float32'),
        np.random.random(size=[1, 144, 24, 36]).astype('float32'),
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