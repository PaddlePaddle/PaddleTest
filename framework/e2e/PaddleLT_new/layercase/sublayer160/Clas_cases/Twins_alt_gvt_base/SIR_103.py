# api:paddle.nn.functional.common.linear||method:reshape||method:transpose||method:transpose||method:reshape||api:paddle.nn.functional.conv._conv_nd||method:reshape||method:transpose||api:paddle.nn.functional.norm.layer_norm||api:paddle.nn.functional.common.linear||method:reshape||method:transpose||method:__getitem__||method:__getitem__||method:transpose
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[192, 192, 4, 4],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[192, 192],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[192],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[192, 384],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[192],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[384],
           dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
           shape=[192],
           dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
           shape=[192],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [11, 784, 192], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = paddle.nn.functional.common.linear(x=var_0, weight=self.parameter_1, bias=self.parameter_4, name=None)
        var_2 = var_1.reshape([11, 784, 6, 32])
        var_3 = var_2.transpose([0, 2, 1, 3])
        var_4 = var_0.transpose([0, 2, 1])
        var_5 = var_4.reshape([11, 192, 28, 28])
        var_6 = paddle.nn.functional.conv._conv_nd(var_5, self.parameter_0, bias=self.parameter_2, stride=[4, 4], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_7 = var_6.reshape([11, 192, 49])
        var_8 = var_7.transpose([0, 2, 1])
        var_9 = paddle.nn.functional.norm.layer_norm(var_8, normalized_shape=[192], weight=self.parameter_7, bias=self.parameter_6, epsilon=1e-05)
        var_10 = paddle.nn.functional.common.linear(x=var_9, weight=self.parameter_3, bias=self.parameter_5, name=None)
        var_11 = var_10.reshape([11, 49, 2, 6, 32])
        var_12 = var_11.transpose([2, 0, 3, 1, 4])
        var_13 = var_12.__getitem__(0)
        var_14 = var_12.__getitem__(1)
        var_15 = var_13.transpose([0, 1, 3, 2])
        return var_3, var_15, var_14


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[11, 784, 192], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[11, 784, 192]).astype('float32'),
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