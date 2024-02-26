# api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.attribute.shape||method:__getitem__||method:expand||method:flatten||method:transpose||api:paddle.tensor.manipulation.concat||api:paddle.tensor.attribute.shape||method:__getitem__||method:__eq__
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[1, 1, 384],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[384, 3, 16, 16],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[384],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 3, 512, 512], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_1, bias=self.parameter_2, stride=[16, 16], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_2 = paddle.tensor.attribute.shape(var_1)
        var_3 = var_2.__getitem__(0)
        var_4 = self.parameter_0.expand((var_3, -1, -1,))
        var_5 = var_1.flatten(2)
        var_6 = var_5.transpose([0, 2, 1])
        var_7 = paddle.tensor.manipulation.concat([var_4, var_6], axis=1)
        var_8 = paddle.tensor.attribute.shape(var_7)
        var_9 = var_8.__getitem__(1)
        var_10 = var_9.__eq__(1025)
        return var_10, var_7, var_2


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 3, 512, 512], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 3, 512, 512]).astype('float32'),
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