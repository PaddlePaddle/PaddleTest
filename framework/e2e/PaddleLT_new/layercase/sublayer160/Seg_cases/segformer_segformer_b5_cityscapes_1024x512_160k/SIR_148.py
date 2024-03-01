# api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:transpose||method:reshape||api:paddle.nn.functional.conv._conv_nd||method:flatten||method:transpose
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[1280, 1, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[1280],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 2048, 1280], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [], dtype: paddle.int32, stop_gradient: True)
        var_2,    # (shape: [], dtype: paddle.int32, stop_gradient: True)
    ):
        var_3 = paddle.tensor.attribute.shape(var_0)
        var_4 = var_3.__getitem__(0)
        var_5 = var_3.__getitem__(1)
        var_6 = var_0.transpose([0, 2, 1])
        var_7 = var_6.reshape([var_4, 1280, var_1, var_2])
        var_8 = paddle.nn.functional.conv._conv_nd(var_7, self.parameter_0, bias=self.parameter_1, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1280, data_format='NCHW', channel_dim=1, op_type='depthwise_conv2d', use_cudnn=False)
        var_9 = var_8.flatten(2)
        var_10 = var_9.transpose([0, 2, 1])
        return var_10


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 2048, 1280], dtype=paddle.float32),
        paddle.randint(low=0, high=10, shape=[1], dtype=paddle.int32),
        paddle.randint(low=0, high=10, shape=[1], dtype=paddle.int32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 2048, 1280]).astype('float32'),
        np.random.randint(low=0, high=10, size=[1], dtype='int32'),
        np.random.randint(low=0, high=10, size=[1], dtype='int32'),
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