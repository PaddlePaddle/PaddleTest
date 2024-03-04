# api:paddle.tensor.manipulation.reshape||api:paddle.tensor.math.clip||method:__rmul__||method:__sub__||api:paddle.nn.functional.vision.grid_sample
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [10, 3200, 2], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [10, 3, 64, 256], dtype: paddle.float32, stop_gradient: True)
    ):
        var_2 = paddle.tensor.manipulation.reshape(var_0, shape=[-1, 32, 100, 2])
        var_3 = paddle.tensor.math.clip(var_2, 0, 1)
        var_4 = var_3.__rmul__(2.0)
        var_5 = var_4.__sub__(1.0)
        var_6 = paddle.nn.functional.vision.grid_sample(var_1, var_5)
        return var_6


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[10, 3200, 2], dtype=paddle.float32),
        paddle.rand(shape=[10, 3, 64, 256], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[10, 3200, 2]).astype('float32'),
        np.random.random(size=[10, 3, 64, 256]).astype('float32'),
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