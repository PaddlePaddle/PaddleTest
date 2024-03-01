# api:paddle.tensor.math.clip||api:paddle.tensor.ops.exp||api:paddle.tensor.ops.exp||method:__rmul__||method:__sub__||method:__rmul__||method:__sub__||method:__rmul__||method:__add__||method:__rmul__||method:__add__||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.concat
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [2, 1, 1], dtype: paddle.float32, stop_gradient: True)
        var_1,    # (shape: [2, 1, 1], dtype: paddle.float32, stop_gradient: True)
        var_2,    # (shape: [2, 1, 1], dtype: paddle.float32, stop_gradient: True)
        var_3,    # (shape: [2, 1, 1], dtype: paddle.float32, stop_gradient: True)
    ):
        var_4 = paddle.tensor.math.clip(var_0, -10000000000.0, 4.135166556742356)
        var_5 = paddle.tensor.ops.exp(var_3)
        var_6 = paddle.tensor.ops.exp(var_4)
        var_7 = var_5.__rmul__(0.5)
        var_8 = var_1.__sub__(var_7)
        var_9 = var_6.__rmul__(0.5)
        var_10 = var_2.__sub__(var_9)
        var_11 = var_5.__rmul__(0.5)
        var_12 = var_1.__add__(var_11)
        var_13 = var_6.__rmul__(0.5)
        var_14 = var_2.__add__(var_13)
        var_15 = paddle.tensor.manipulation.reshape(var_8, shape=(-1,))
        var_16 = paddle.tensor.manipulation.reshape(var_10, shape=(-1,))
        var_17 = paddle.tensor.manipulation.reshape(var_12, shape=(-1,))
        var_18 = paddle.tensor.manipulation.reshape(var_14, shape=(-1,))
        var_19 = paddle.tensor.manipulation.concat([var_15, var_16, var_17, var_18])
        return var_19


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[2, 1, 1], dtype=paddle.float32),
        paddle.rand(shape=[2, 1, 1], dtype=paddle.float32),
        paddle.rand(shape=[2, 1, 1], dtype=paddle.float32),
        paddle.rand(shape=[2, 1, 1], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[2, 1, 1]).astype('float32'),
        np.random.random(size=[2, 1, 1]).astype('float32'),
        np.random.random(size=[2, 1, 1]).astype('float32'),
        np.random.random(size=[2, 1, 1]).astype('float32'),
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