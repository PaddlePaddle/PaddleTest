# api:paddle.nn.functional.common.interpolate||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.common.dropout2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.common.interpolate||api:paddle.nn.functional.common.interpolate||api:paddle.nn.functional.common.interpolate||api:paddle.nn.functional.common.interpolate
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[19, 51, 1, 1],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 19, 128, 256], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 32, 256, 512], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [1, 19, 64, 128], dtype: paddle.float32, stop_gradient: False)
    ):
        var_3 = paddle.nn.functional.common.interpolate(var_0, scale_factor=2, mode='bilinear', align_corners=True)
        var_4 = paddle.tensor.manipulation.concat([var_1, var_3], axis=1)
        var_5 = paddle.nn.functional.common.dropout2d(var_4, p=0.0, training=True, data_format='NCHW', name=None)
        var_6 = paddle.nn.functional.conv._conv_nd(var_5, self.parameter_0, bias=None, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_7 = paddle.nn.functional.common.interpolate(var_6, scale_factor=2, mode='bilinear', align_corners=True)
        var_8 = paddle.nn.functional.common.interpolate(var_2, scale_factor=2, mode='bilinear', align_corners=True)
        var_9 = paddle.nn.functional.common.interpolate(var_8, scale_factor=2, mode='bilinear', align_corners=True)
        var_10 = paddle.nn.functional.common.interpolate(var_9, scale_factor=2, mode='bilinear', align_corners=True)
        return var_7, var_10



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, 19, -1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, 32, -1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[1, 19, 128, 256], dtype=paddle.float32),
        paddle.rand(shape=[1, 32, 256, 512], dtype=paddle.float32),
        paddle.rand(shape=[1, 19, 64, 128], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 19, 128, 256]).astype('float32'),
        np.random.random(size=[1, 32, 256, 512]).astype('float32'),
        np.random.random(size=[1, 19, 64, 128]).astype('float32'),
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