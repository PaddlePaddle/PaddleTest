# api:paddle.nn.functional.conv._conv_nd||method:__mul__||method:__mul__||api:paddle.nn.functional.conv._conv_nd||method:__mul__||method:__add__||api:paddle.nn.functional.activation.elu||method:__mul__||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.math.divide||api:paddle.tensor.math.sign||api:paddle.tensor.layer_function_generator.abs||api:paddle.tensor.ops.floor||api:paddle.tensor.math.multiply||method:__mul__||method:__sub__||api:paddle.tensor.manipulation.concat||method:flatten||method:transpose
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[2, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[1, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[2],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[2, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[1],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[1],
           dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
           shape=[2],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 256, 8, 8], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1], dtype: paddle.float32, stop_gradient: True)
    ):
        var_2 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_0, bias=self.parameter_2, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_3 = var_2.__mul__(self.parameter_5)
        var_4 = var_3.__mul__(128)
        var_5 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_3, bias=self.parameter_6, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_6 = var_5.__mul__(self.parameter_5)
        var_7 = var_6.__add__(1.0)
        var_8 = paddle.nn.functional.activation.elu(var_7)
        var_9 = var_8.__mul__(128)
        var_10 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_1, bias=self.parameter_4, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_11 = paddle.tensor.math.divide(var_10, var_1)
        var_12 = paddle.tensor.math.sign(var_11)
        var_13 = paddle.tensor.abs(var_11)
        var_14 = paddle.tensor.ops.floor(var_13)
        var_15 = paddle.tensor.math.multiply(var_12, var_14)
        var_16 = var_15.__mul__(var_1)
        var_17 = var_10.__sub__(var_16)
        var_18 = paddle.tensor.manipulation.concat([var_4, var_9, var_17], axis=1)
        var_19 = var_18.flatten(2)
        var_20 = var_19.transpose((0, 2, 1,))
        return var_4, var_9, var_17, var_18, var_20


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 256, 8, 8], dtype=paddle.float32),
        paddle.rand(shape=[1], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 256, 8, 8]).astype('float32'),
        np.random.random(size=[1]).astype('float32'),
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