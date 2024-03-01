# method:unsqueeze||api:paddle.tensor.search.masked_select||method:sum||method:__mul__||method:__ne__||method:astype||method:sum||method:clip||method:__add__||method:__truediv__
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [1, 2434], dtype: paddle.bool, stop_gradient: True)
        var_1,    # (shape: [1, 2434, 1], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [1, 2434, 1], dtype: paddle.int64, stop_gradient: True)
        var_3,    # (shape: [], dtype: paddle.float32, stop_gradient: False)
    ):
        var_4 = var_0.unsqueeze(-1)
        var_5 = paddle.tensor.search.masked_select(var_1, var_4)
        var_6 = var_5.sum()
        var_7 = var_6.__mul__(1.0)
        var_8 = var_2.__ne__(80)
        var_9 = var_8.astype('float32')
        var_10 = var_9.sum()
        var_11 = var_10.clip(min=1)
        var_12 = var_7.__add__(var_3)
        var_13 = var_12.__truediv__(var_11)
        return var_13


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 2434], dtype=paddle.bool),
        paddle.rand(shape=[1, 2434, 1], dtype=paddle.float32),
        paddle.randint(low=0, high=10, shape=[1, 2434, 1], dtype=paddle.int64),
        paddle.rand(shape=[1], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 2434]).astype('bool'),
        np.random.random(size=[1, 2434, 1]).astype('float32'),
        np.random.randint(low=0, high=10, size=[1, 2434, 1], dtype='int64'),
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