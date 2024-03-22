# method:chunk||method:chunk||method:chunk
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [1, 40, 128, 256], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 80, 64, 128], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [1, 160, 32, 64], dtype: paddle.float32, stop_gradient: False)
    ):
        out = var_0.chunk(2, axis=1)
        var_3 = out[0]
        var_4 = out[1]
        out = var_1.chunk(2, axis=1)
        var_5 = out[0]
        var_6 = out[1]
        out = var_2.chunk(2, axis=1)
        var_7 = out[0]
        var_8 = out[1]
        return var_4, var_6, var_8, var_3, var_5, var_7


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 40, 128, 256], dtype=paddle.float32),
        paddle.rand(shape=[1, 80, 64, 128], dtype=paddle.float32),
        paddle.rand(shape=[1, 160, 32, 64], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 40, 128, 256]).astype('float32'),
        np.random.random(size=[1, 80, 64, 128]).astype('float32'),
        np.random.random(size=[1, 160, 32, 64]).astype('float32'),
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