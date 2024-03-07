# api:paddle.tensor.attribute.shape||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.bmm||api:paddle.nn.functional.activation.softmax||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.bmm||method:__getitem__||method:__getitem__||api:paddle.tensor.manipulation.reshape||method:__mul__||method:__add__
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[512],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[512, 512, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[64, 512, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[64, 512, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[64],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[1],
           dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
           shape=[64],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 512, 64, 128], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = paddle.tensor.attribute.shape(var_0)
        var_2 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_3, bias=self.parameter_4, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_3 = paddle.tensor.manipulation.reshape(var_2, (0, 64, -1,))
        var_4 = paddle.tensor.linalg.transpose(var_3, (0, 2, 1,))
        var_5 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_2, bias=self.parameter_6, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_6 = paddle.tensor.manipulation.reshape(var_5, (0, 64, -1,))
        var_7 = paddle.tensor.linalg.bmm(var_4, var_6)
        var_8 = paddle.nn.functional.activation.softmax(var_7, axis=-1)
        var_9 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_1, bias=self.parameter_0, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_10 = paddle.tensor.manipulation.reshape(var_9, (0, 512, -1,))
        var_11 = paddle.tensor.linalg.transpose(var_8, (0, 2, 1,))
        var_12 = paddle.tensor.linalg.bmm(var_10, var_11)
        var_13 = var_1.__getitem__(2)
        var_14 = var_1.__getitem__(3)
        var_15 = paddle.tensor.manipulation.reshape(var_12, (0, 512, var_13, var_14,))
        var_16 = self.parameter_5.__mul__(var_15)
        var_17 = var_16.__add__(var_0)
        return var_17


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 512, 64, 128], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 512, 64, 128]).astype('float32'),
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