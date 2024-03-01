# api:paddle.tensor.manipulation.split||method:__mul__||method:__add__||api:paddle.nn.functional.activation.elu||method:__add__||method:__mul__||method:reshape||api:paddle.nn.functional.activation.softmax||method:matmul||api:paddle.tensor.manipulation.concat||method:detach||method:detach
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [1, 21504, 15], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 21504, 4], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [1, 21504, 91], dtype: paddle.float32, stop_gradient: False)
        var_3,    # (shape: [1, 21504, 2], dtype: paddle.float32, stop_gradient: True)
        var_4,    # (shape: [1, 21504, 1], dtype: paddle.float32, stop_gradient: True)
        var_5,    # (shape: [91], dtype: paddle.float32, stop_gradient: True)
    ):
        out = paddle.tensor.manipulation.split(var_1, 2, axis=-1)
        var_6 = out[0]
        var_7 = out[1]
        var_8 = var_6.__mul__(var_4)
        var_9 = var_8.__add__(var_3)
        var_10 = paddle.nn.functional.activation.elu(var_7)
        var_11 = var_10.__add__(1.0)
        var_12 = var_11.__mul__(var_4)
        var_13 = var_2.reshape([1, 21504, 1, 91])
        var_14 = paddle.nn.functional.activation.softmax(var_13)
        var_15 = var_14.matmul(var_5)
        var_16 = paddle.tensor.manipulation.concat([var_9, var_12, var_15], axis=-1)
        var_17 = var_0.detach()
        var_18 = var_16.detach()
        return var_17, var_18, var_16


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 21504, 15], dtype=paddle.float32),
        paddle.rand(shape=[1, 21504, 4], dtype=paddle.float32),
        paddle.rand(shape=[1, 21504, 91], dtype=paddle.float32),
        paddle.rand(shape=[1, 21504, 2], dtype=paddle.float32),
        paddle.rand(shape=[1, 21504, 1], dtype=paddle.float32),
        paddle.rand(shape=[91], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 21504, 15]).astype('float32'),
        np.random.random(size=[1, 21504, 4]).astype('float32'),
        np.random.random(size=[1, 21504, 91]).astype('float32'),
        np.random.random(size=[1, 21504, 2]).astype('float32'),
        np.random.random(size=[1, 21504, 1]).astype('float32'),
        np.random.random(size=[91]).astype('float32'),
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