# method:transpose||method:reshape||api:paddle.nn.functional.conv._conv_nd||method:__add__||method:flatten||method:transpose
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[768],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[768, 1, 3, 3],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [43, 49, 768], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = var_0.transpose([0, 2, 1])
        var_2 = var_1.reshape([43, 768, 7, 7])
        var_3 = paddle.nn.functional.conv._conv_nd(var_2, self.parameter_1, bias=self.parameter_0, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=768, data_format='NCHW', channel_dim=1, op_type='depthwise_conv2d', use_cudnn=False)
        var_4 = var_3.__add__(var_2)
        var_5 = var_4.flatten(2)
        var_6 = var_5.transpose([0, 2, 1])
        return var_6


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[43, 49, 768], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[43, 49, 768]).astype('float32'),
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