# method:reshape||api:paddle.nn.functional.activation.softmax||method:topk||method:mean||api:paddle.tensor.manipulation.concat||method:reshape||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.ops.sigmoid
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[1, 64, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[64, 20, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[1],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[64],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 68, 7, 10], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = var_0.reshape([1, 4, -1, 7, 10])
        var_2 = paddle.nn.functional.activation.softmax(var_1, axis=2)
        out = var_2.topk(4, axis=2)
        var_3 = out[0]
        var_4 = out[1]
        var_5 = var_3.mean(axis=2, keepdim=True)
        var_6 = paddle.tensor.manipulation.concat([var_3, var_5], axis=2)
        var_7 = var_6.reshape([1, 20, 7, 10])
        var_8 = paddle.nn.functional.conv._conv_nd(var_7, self.parameter_1, bias=self.parameter_3, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_9 = paddle.nn.functional.activation.relu(var_8)
        var_10 = paddle.nn.functional.conv._conv_nd(var_9, self.parameter_0, bias=self.parameter_2, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_11 = paddle.tensor.ops.sigmoid(var_10)
        return var_11


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 68, 7, 10], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 68, 7, 10]).astype('float32'),
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