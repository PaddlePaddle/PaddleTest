# api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.manipulation.split||api:paddle.tensor.ops.sigmoid||api:paddle.vision.ops.deform_conv2d
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[27],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[128],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[27, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[128, 256, 3, 3],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 256, 50, 76], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_2, bias=self.parameter_0, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        out = paddle.tensor.manipulation.split(var_1, num_or_sections=[18, 9], axis=1)
        var_2 = out[0]
        var_3 = out[1]
        var_4 = paddle.tensor.ops.sigmoid(var_3)
        var_5 = paddle.vision.ops.deform_conv2d(x=var_0, offset=var_2, weight=self.parameter_3, bias=self.parameter_1, stride=[1, 1], padding=[1, 1], dilation=[1, 1], deformable_groups=1, groups=1, mask=var_4)
        return var_5


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 256, 50, 76], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 256, 50, 76]).astype('float32'),
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