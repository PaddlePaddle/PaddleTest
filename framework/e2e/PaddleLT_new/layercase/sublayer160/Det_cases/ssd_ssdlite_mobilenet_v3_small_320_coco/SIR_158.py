# api:paddle.tensor.search.masked_select||api:paddle.tensor.search.masked_select||api:paddle.nn.functional.loss.smooth_l1_loss||method:__mul__||api:paddle.nn.functional.loss.cross_entropy||method:squeeze||method:squeeze
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [1, 2434, 4], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 2434, 4], dtype: paddle.bool, stop_gradient: True)
        var_2,    # (shape: [1, 2434, 4], dtype: paddle.float32, stop_gradient: True)
        var_3,    # (shape: [1, 2434, 81], dtype: paddle.float32, stop_gradient: False)
        var_4,    # (shape: [1, 2434, 1], dtype: paddle.int64, stop_gradient: True)
    ):
        var_5 = paddle.tensor.search.masked_select(var_0, var_1)
        var_6 = paddle.tensor.search.masked_select(var_2, var_1)
        var_7 = paddle.nn.functional.loss.smooth_l1_loss(var_5, var_6, reduction='sum')
        var_8 = var_7.__mul__(1.0)
        var_9 = paddle.nn.functional.loss.cross_entropy(var_3, var_4, reduction='none')
        var_10 = var_9.squeeze(-1)
        var_11 = var_4.squeeze(-1)
        return var_10, var_11, var_9, var_8


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 2434, 4], dtype=paddle.float32),
        paddle.rand(shape=[1, 2434, 4], dtype=paddle.bool),
        paddle.rand(shape=[1, 2434, 4], dtype=paddle.float32),
        paddle.rand(shape=[1, 2434, 81], dtype=paddle.float32),
        paddle.randint(low=0, high=10, shape=[1, 2434, 1], dtype=paddle.int64),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 2434, 4]).astype('float32'),
        np.random.random(size=[1, 2434, 4]).astype('bool'),
        np.random.random(size=[1, 2434, 4]).astype('float32'),
        np.random.random(size=[1, 2434, 81]).astype('float32'),
        np.random.randint(low=0, high=10, size=[1, 2434, 1], dtype='int64'),
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