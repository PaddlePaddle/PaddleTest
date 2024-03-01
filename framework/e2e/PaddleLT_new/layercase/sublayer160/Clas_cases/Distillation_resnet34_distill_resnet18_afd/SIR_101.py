# method:pow||method:mean||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||method:reshape||method:pow||method:mean||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||method:reshape||method:pow||method:mean||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||method:reshape||method:pow||method:mean||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||method:reshape||method:pow||method:mean||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||method:reshape||method:pow||method:mean||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||method:reshape||method:pow||method:mean||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||method:reshape||method:pow||method:mean||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||method:reshape||api:paddle.tensor.manipulation.stack
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [10, 64, 56, 56], dtype: paddle.float32, stop_gradient: True)
        var_1,    # (shape: [10, 64, 56, 56], dtype: paddle.float32, stop_gradient: True)
        var_2,    # (shape: [10, 128, 28, 28], dtype: paddle.float32, stop_gradient: True)
        var_3,    # (shape: [10, 128, 28, 28], dtype: paddle.float32, stop_gradient: True)
        var_4,    # (shape: [10, 256, 14, 14], dtype: paddle.float32, stop_gradient: True)
        var_5,    # (shape: [10, 256, 14, 14], dtype: paddle.float32, stop_gradient: True)
        var_6,    # (shape: [10, 512, 7, 7], dtype: paddle.float32, stop_gradient: True)
        var_7,    # (shape: [10, 512, 7, 7], dtype: paddle.float32, stop_gradient: True)
    ):
        var_8 = var_0.pow(2)
        var_9 = var_8.mean(1, keepdim=True)
        var_10 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_9, output_size=(56, 56,), data_format='NCHW', name=None)
        var_11 = var_10.reshape([10, 3136])
        var_12 = var_1.pow(2)
        var_13 = var_12.mean(1, keepdim=True)
        var_14 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_13, output_size=(56, 56,), data_format='NCHW', name=None)
        var_15 = var_14.reshape([10, 3136])
        var_16 = var_2.pow(2)
        var_17 = var_16.mean(1, keepdim=True)
        var_18 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_17, output_size=(56, 56,), data_format='NCHW', name=None)
        var_19 = var_18.reshape([10, 3136])
        var_20 = var_3.pow(2)
        var_21 = var_20.mean(1, keepdim=True)
        var_22 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_21, output_size=(56, 56,), data_format='NCHW', name=None)
        var_23 = var_22.reshape([10, 3136])
        var_24 = var_4.pow(2)
        var_25 = var_24.mean(1, keepdim=True)
        var_26 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_25, output_size=(56, 56,), data_format='NCHW', name=None)
        var_27 = var_26.reshape([10, 3136])
        var_28 = var_5.pow(2)
        var_29 = var_28.mean(1, keepdim=True)
        var_30 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_29, output_size=(56, 56,), data_format='NCHW', name=None)
        var_31 = var_30.reshape([10, 3136])
        var_32 = var_6.pow(2)
        var_33 = var_32.mean(1, keepdim=True)
        var_34 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_33, output_size=(56, 56,), data_format='NCHW', name=None)
        var_35 = var_34.reshape([10, 3136])
        var_36 = var_7.pow(2)
        var_37 = var_36.mean(1, keepdim=True)
        var_38 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_37, output_size=(56, 56,), data_format='NCHW', name=None)
        var_39 = var_38.reshape([10, 3136])
        var_40 = paddle.tensor.manipulation.stack([var_11, var_15, var_19, var_23, var_27, var_31, var_35, var_39], axis=1)
        return var_40


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[10, 64, 56, 56], dtype=paddle.float32),
        paddle.rand(shape=[10, 64, 56, 56], dtype=paddle.float32),
        paddle.rand(shape=[10, 128, 28, 28], dtype=paddle.float32),
        paddle.rand(shape=[10, 128, 28, 28], dtype=paddle.float32),
        paddle.rand(shape=[10, 256, 14, 14], dtype=paddle.float32),
        paddle.rand(shape=[10, 256, 14, 14], dtype=paddle.float32),
        paddle.rand(shape=[10, 512, 7, 7], dtype=paddle.float32),
        paddle.rand(shape=[10, 512, 7, 7], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[10, 64, 56, 56]).astype('float32'),
        np.random.random(size=[10, 64, 56, 56]).astype('float32'),
        np.random.random(size=[10, 128, 28, 28]).astype('float32'),
        np.random.random(size=[10, 128, 28, 28]).astype('float32'),
        np.random.random(size=[10, 256, 14, 14]).astype('float32'),
        np.random.random(size=[10, 256, 14, 14]).astype('float32'),
        np.random.random(size=[10, 512, 7, 7]).astype('float32'),
        np.random.random(size=[10, 512, 7, 7]).astype('float32'),
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