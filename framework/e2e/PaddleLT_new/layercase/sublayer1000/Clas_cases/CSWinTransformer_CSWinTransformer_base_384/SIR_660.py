# method:__add__||method:transpose||method:reshape||method:reshape||method:transpose||method:reshape
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [20, 8, 288, 24], dtype: paddle.float32, stop_gradient: True)
        var_1,    # (shape: [20, 8, 288, 24], dtype: paddle.float32, stop_gradient: True)
    ):
        var_2 = var_0.__add__(var_1)
        var_3 = var_2.transpose([0, 2, 1, 3])
        var_4 = var_3.reshape([-1, 288, 192])
        var_5 = var_0.shape[0] / 2
        var_6 = var_4.reshape([var_5, 1, 2, 24, 12, 192])
        var_7 = var_6.transpose([0, 1, 3, 2, 4, 5])
        var_8 = var_7.reshape([var_5, 24, 24, 192])
        return var_8


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[20, 8, 288, 24], dtype=paddle.float32),
        paddle.rand(shape=[20, 8, 288, 24], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[20, 8, 288, 24]).astype('float32'),
        np.random.random(size=[20, 8, 288, 24]).astype('float32'),
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