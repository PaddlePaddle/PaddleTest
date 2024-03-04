# api:paddle.nn.functional.norm.layer_norm||method:reshape||method:transpose||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:flatten||method:transpose||api:paddle.nn.functional.norm.layer_norm
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
           shape=[512],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[512],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[320],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[512, 320, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[320],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 4096, 320], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [], dtype: paddle.int32, stop_gradient: True)
        var_2,    # (shape: [], dtype: paddle.int32, stop_gradient: True)
        var_3,    # (shape: [], dtype: paddle.int32, stop_gradient: True)
    ):
        var_4 = paddle.nn.functional.norm.layer_norm(var_0, normalized_shape=[320], weight=self.parameter_5, bias=self.parameter_3, epsilon=1e-06)
        var_5 = var_4.reshape([var_1, var_2, var_3, 320])
        var_6 = var_5.transpose([0, 3, 1, 2])
        var_7 = paddle.nn.functional.conv._conv_nd(var_6, self.parameter_4, bias=self.parameter_2, stride=[2, 2], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_8 = paddle.tensor.attribute.shape(var_7)
        var_9 = var_8.__getitem__(2)
        var_10 = var_8.__getitem__(3)
        var_11 = var_7.flatten(2)
        var_12 = var_11.transpose([0, 2, 1])
        var_13 = paddle.nn.functional.norm.layer_norm(var_12, normalized_shape=[512], weight=self.parameter_1, bias=self.parameter_0, epsilon=1e-05)
        return var_13, var_9, var_10, var_6


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 4096, 320], dtype=paddle.float32),
        paddle.randint(low=0, high=10, shape=[1], dtype=paddle.int32),
        paddle.randint(low=0, high=10, shape=[1], dtype=paddle.int32),
        paddle.randint(low=0, high=10, shape=[1], dtype=paddle.int32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 4096, 320]).astype('float32'),
        np.random.randint(low=0, high=10, size=[1], dtype='int32'),
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