# api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.split||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.transpose
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [22, 196, 640], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = paddle.tensor.manipulation.reshape(var_0, [22, 196, 8, -1])
        out = paddle.tensor.manipulation.split(var_1, [16, 64], axis=3)
        var_2 = out[0]
        var_3 = out[1]
        var_4 = paddle.tensor.linalg.transpose(var_2, perm=[0, 2, 1, 3])
        var_5 = paddle.tensor.linalg.transpose(var_3, perm=[0, 2, 1, 3])
        return var_4, var_5


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[22, 196, 640], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[22, 196, 640]).astype('float32'),
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