# api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.slice||method:__mul__||api:paddle.tensor.manipulation.slice||method:__mul__||api:paddle.tensor.manipulation.slice||method:__mul__||api:paddle.tensor.manipulation.slice||method:__mul__
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [2, 4], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = paddle.tensor.manipulation.reshape(var_0, shape=(0, -1, 4,))
        var_2 = paddle.tensor.manipulation.slice(var_1, axes=[2], starts=[0], ends=[1])
        var_3 = var_2.__mul__(0.1)
        var_4 = paddle.tensor.manipulation.slice(var_1, axes=[2], starts=[1], ends=[2])
        var_5 = var_4.__mul__(0.1)
        var_6 = paddle.tensor.manipulation.slice(var_1, axes=[2], starts=[2], ends=[3])
        var_7 = var_6.__mul__(0.2)
        var_8 = paddle.tensor.manipulation.slice(var_1, axes=[2], starts=[3], ends=[4])
        var_9 = var_8.__mul__(0.2)
        return var_7, var_9, var_3, var_5


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[2, 4], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[2, 4]).astype('float32'),
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