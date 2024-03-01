# api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||method:flatten||method:flatten
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
    ):
        var_0 = paddle.tensor.creation.arange(22, dtype='float32')
        var_1 = var_0.__add__(0.5)
        var_2 = var_1.__mul__(16)
        var_3 = paddle.tensor.creation.arange(22, dtype='float32')
        var_4 = var_3.__add__(0.5)
        var_5 = var_4.__mul__(16)
        out = paddle.tensor.creation.meshgrid(var_5, var_2)
        var_6 = out[0]
        var_7 = out[1]
        var_8 = var_6.flatten()
        var_9 = var_7.flatten()
        return var_8, var_9


def create_paddle_inputs():
    inputs = (
    )
    return inputs


def create_numpy_inputs():
    inputs = (
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