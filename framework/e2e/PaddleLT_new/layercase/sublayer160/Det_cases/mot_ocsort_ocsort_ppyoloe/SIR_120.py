# api:paddle.nn.functional.pooling.max_pool2d||api:paddle.nn.functional.pooling.max_pool2d||api:paddle.nn.functional.pooling.max_pool2d||api:paddle.tensor.manipulation.concat
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [1, 384, 17, 17], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = paddle.nn.functional.pooling.max_pool2d(var_0, kernel_size=5, stride=1, padding=2, return_mask=False, ceil_mode=False, data_format='NCHW', name=None)
        var_2 = paddle.nn.functional.pooling.max_pool2d(var_0, kernel_size=9, stride=1, padding=4, return_mask=False, ceil_mode=False, data_format='NCHW', name=None)
        var_3 = paddle.nn.functional.pooling.max_pool2d(var_0, kernel_size=13, stride=1, padding=6, return_mask=False, ceil_mode=False, data_format='NCHW', name=None)
        var_4 = paddle.tensor.manipulation.concat([var_0, var_1, var_2, var_3], axis=1)
        return var_4


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 384, 17, 17], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 384, 17, 17]).astype('float32'),
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