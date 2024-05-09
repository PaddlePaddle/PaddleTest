# api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||method:__add__||method:__add__||method:__add__||method:__truediv__
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[19, 32, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[19, 32, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[19],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[19],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[19],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[19],
           dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
           shape=[19, 32, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
           shape=[19, 32, 1, 1],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 32, 512, 1024], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 32, 512, 1024], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [1, 32, 512, 1024], dtype: paddle.float32, stop_gradient: False)
        var_3,    # (shape: [1, 32, 512, 1024], dtype: paddle.float32, stop_gradient: False)
    ):
        var_4 = paddle.nn.functional.conv._conv_nd(var_1, self.parameter_1, bias=self.parameter_5, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_5 = paddle.nn.functional.conv._conv_nd(var_2, self.parameter_7, bias=self.parameter_3, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_6 = paddle.nn.functional.conv._conv_nd(var_3, self.parameter_0, bias=self.parameter_4, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_7 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_6, bias=self.parameter_2, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_8 = var_4.__add__(var_5)
        var_9 = var_8.__add__(var_6)
        var_10 = var_9.__add__(var_7)
        var_11 = var_10.__truediv__(4)
        return var_11


def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[1, 32, 512, 1024], dtype=paddle.float32),
        paddle.rand(shape=[1, 32, 512, 1024], dtype=paddle.float32),
        paddle.rand(shape=[1, 32, 512, 1024], dtype=paddle.float32),
        paddle.rand(shape=[1, 32, 512, 1024], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 32, 512, 1024]).astype('float32'),
        np.random.random(size=[1, 32, 512, 1024]).astype('float32'),
        np.random.random(size=[1, 32, 512, 1024]).astype('float32'),
        np.random.random(size=[1, 32, 512, 1024]).astype('float32'),
    )
    return inputs


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = create_tensor_inputs()
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