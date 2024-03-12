# api:paddle.tensor.manipulation.concat||api:paddle.tensor.attribute.shape||method:__getitem__||api:paddle.tensor.attribute.shape||method:__getitem__||api:paddle.tensor.attribute.shape||method:__getitem__||api:paddle.tensor.attribute.shape||method:__getitem__||api:paddle.tensor.manipulation.reshape
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [1, 256, 32, 64], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 256, 32, 64], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [1, 3, 512, 1024], dtype: paddle.float32, stop_gradient: True)
    ):
        var_3 = paddle.tensor.manipulation.concat([var_1, var_0], axis=1)
        var_4 = paddle.tensor.attribute.shape(var_1)
        var_5 = var_4.__getitem__(2)
        var_6 = paddle.tensor.attribute.shape(var_2)
        var_7 = var_6.__getitem__(2)
        var_8 = paddle.tensor.attribute.shape(var_2)
        var_9 = var_8.__getitem__(2)
        var_10 = paddle.tensor.attribute.shape(var_2)
        var_11 = var_10.__getitem__(3)
        var_12 = paddle.tensor.manipulation.reshape(var_2, [1, 3, var_9, var_11])
        return var_3, var_12, var_7, var_5


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 256, 32, 64], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 32, 64], dtype=paddle.float32),
        paddle.rand(shape=[1, 3, 512, 1024], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 256, 32, 64]).astype('float32'),
        np.random.random(size=[1, 256, 32, 64]).astype('float32'),
        np.random.random(size=[1, 3, 512, 1024]).astype('float32'),
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