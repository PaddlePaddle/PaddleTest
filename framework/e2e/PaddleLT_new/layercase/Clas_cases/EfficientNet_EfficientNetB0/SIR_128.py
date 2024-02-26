# api:paddle.tensor.attribute.shape||method:__getitem__||api:paddle.tensor.random.rand||method:__radd__||api:paddle.tensor.ops.floor||api:paddle.tensor.math.multiply||method:__truediv__||api:paddle.tensor.math.add
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [11, 112, 14, 14], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [11, 112, 14, 14], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = paddle.tensor.attribute.shape(var_0)
        var_3 = var_2.__getitem__(0)
        var_4 = paddle.tensor.random.rand(shape=[var_3, 1, 1, 1])
        var_5 = var_4.__radd__(0.875)
        var_6 = paddle.tensor.ops.floor(var_5)
        var_7 = paddle.tensor.math.multiply(var_0, var_6)
        var_8 = var_7.__truediv__(0.875)
        var_9 = paddle.tensor.math.add(var_8, var_1)
        return var_9


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[11, 112, 14, 14], dtype=paddle.float32),
        paddle.rand(shape=[11, 112, 14, 14], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[11, 112, 14, 14]).astype('float32'),
        np.random.random(size=[11, 112, 14, 14]).astype('float32'),
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