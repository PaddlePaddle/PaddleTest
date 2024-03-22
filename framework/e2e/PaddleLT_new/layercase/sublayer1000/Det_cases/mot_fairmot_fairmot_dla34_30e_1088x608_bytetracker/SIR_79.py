# method:__neg__||api:paddle.tensor.ops.exp||method:__mul__||method:__neg__||api:paddle.tensor.ops.exp||method:__mul__||method:__add__||method:__add__||method:__add__||method:__mul__
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[1],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[1],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = self.parameter_0.__neg__()
        var_3 = paddle.tensor.ops.exp(var_2)
        var_4 = var_3.__mul__(var_1)
        var_5 = self.parameter_1.__neg__()
        var_6 = paddle.tensor.ops.exp(var_5)
        var_7 = var_6.__mul__(var_0)
        var_8 = var_4.__add__(var_7)
        var_9 = self.parameter_0.__add__(self.parameter_1)
        var_10 = var_8.__add__(var_9)
        var_11 = var_10.__mul__(0.5)
        return var_11


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1], dtype=paddle.float32),
        paddle.rand(shape=[1], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
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