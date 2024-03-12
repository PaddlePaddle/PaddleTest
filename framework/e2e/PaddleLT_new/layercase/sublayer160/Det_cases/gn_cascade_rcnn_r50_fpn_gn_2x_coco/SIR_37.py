# api:paddle.tensor.math.add_n
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_3,    # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_4,    # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_5,    # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_6,    # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_7,    # (shape: [], dtype: paddle.float32, stop_gradient: False)
    ):
        var_8 = paddle.tensor.math.add_n([var_6, var_7, var_0, var_1, var_2, var_3, var_4, var_5])
        return var_8


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1], dtype=paddle.float32),
        paddle.rand(shape=[1], dtype=paddle.float32),
        paddle.rand(shape=[1], dtype=paddle.float32),
        paddle.rand(shape=[1], dtype=paddle.float32),
        paddle.rand(shape=[1], dtype=paddle.float32),
        paddle.rand(shape=[1], dtype=paddle.float32),
        paddle.rand(shape=[1], dtype=paddle.float32),
        paddle.rand(shape=[1], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1]).astype('float32'),
        np.random.random(size=[1]).astype('float32'),
        np.random.random(size=[1]).astype('float32'),
        np.random.random(size=[1]).astype('float32'),
        np.random.random(size=[1]).astype('float32'),
        np.random.random(size=[1]).astype('float32'),
        np.random.random(size=[1]).astype('float32'),
        np.random.random(size=[1]).astype('float32'),
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