# method:__sub__||method:pow||api:paddle.nn.functional.loss.binary_cross_entropy||method:__truediv__||method:__rmul__||method:__rmul__||method:__add__
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [1, 21824, 15], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 21824, 15], dtype: paddle.float32, stop_gradient: True)
        var_2,    # (shape: [], dtype: paddle.float32, stop_gradient: False)
    ):
        var_3 = var_0.__sub__(var_1)
        var_4 = var_3.pow(2.0)
        var_5 = paddle.nn.functional.loss.binary_cross_entropy(var_0, var_1, weight=var_4, reduction='sum')
        var_6 = var_5.__truediv__(13)
        var_7 = var_6.__rmul__(1.0)
        var_8 = var_2.__rmul__(1.0)
        var_9 = var_7.__add__(var_8)
        return var_9, var_6


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 21824, 15], dtype=paddle.float32),
        paddle.rand(shape=[1, 21824, 15], dtype=paddle.float32),
        paddle.rand(shape=[1], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 21824, 15]).astype('float32'),
        np.random.random(size=[1, 21824, 15]).astype('float32'),
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