# api:paddle.nn.functional.activation.swish||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.swish||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.ops.sigmoid||api:paddle.tensor.math.multiply
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[8, 32, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[32],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[32, 8, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[8],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [11, 32, 112, 112], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1 = paddle.nn.functional.activation.swish(var_0)
        var_2 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_1, output_size=1, data_format='NCHW', name=None)
        var_3 = paddle.nn.functional.conv._conv_nd(var_2, self.parameter_0, bias=self.parameter_3, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_4 = paddle.nn.functional.activation.swish(var_3)
        var_5 = paddle.nn.functional.conv._conv_nd(var_4, self.parameter_2, bias=self.parameter_1, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_6 = paddle.tensor.ops.sigmoid(var_5)
        var_7 = paddle.tensor.math.multiply(var_1, var_6)
        return var_7



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, 32, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[11, 32, 112, 112], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[11, 32, 112, 112]).astype('float32'),
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