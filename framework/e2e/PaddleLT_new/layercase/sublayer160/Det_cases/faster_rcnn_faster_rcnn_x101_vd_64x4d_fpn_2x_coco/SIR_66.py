# api:paddle.tensor.manipulation.gather||method:__eq__||api:paddle.tensor.creation.ones_like||method:__mul__||api:paddle.tensor.search.where||method:__eq__||api:paddle.tensor.creation.ones_like||method:__mul__||api:paddle.tensor.search.where
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [1002], dtype: paddle.int64, stop_gradient: True)
        var_1,    # (shape: [1002], dtype: paddle.int32, stop_gradient: True)
        var_2,    # (shape: [2], dtype: paddle.int32, stop_gradient: True)
    ):
        var_3 = paddle.tensor.manipulation.gather(var_2, var_0)
        var_4 = var_1.__eq__(0)
        var_5 = paddle.tensor.creation.ones_like(var_3)
        var_6 = var_5.__mul__(80)
        var_7 = paddle.tensor.search.where(var_4, var_6, var_3)
        var_8 = var_1.__eq__(-1)
        var_9 = paddle.tensor.creation.ones_like(var_7)
        var_10 = var_9.__mul__(-1)
        var_11 = paddle.tensor.search.where(var_8, var_10, var_7)
        return var_11


def create_paddle_inputs():
    inputs = (
        paddle.randint(low=0, high=10, shape=[1002], dtype=paddle.int64),
        paddle.randint(low=0, high=10, shape=[1002], dtype=paddle.int32),
        paddle.randint(low=0, high=10, shape=[2], dtype=paddle.int32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.randint(low=0, high=10, size=[1002], dtype='int64'),
        np.random.randint(low=0, high=10, size=[1002], dtype='int32'),
        np.random.randint(low=0, high=10, size=[2], dtype='int32'),
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