# api:paddle.tensor.attribute.shape||method:__getitem__||api:paddle.tensor.creation.linspace||api:paddle.tensor.attribute.shape||method:__getitem__||api:paddle.tensor.creation.linspace||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.unsqueeze||api:paddle.tensor.manipulation.unsqueeze||api:paddle.tensor.attribute.shape||method:__getitem__||api:paddle.tensor.manipulation.expand||api:paddle.tensor.attribute.shape||method:__getitem__||api:paddle.tensor.manipulation.expand||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [1, 256, 25, 38], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = paddle.tensor.attribute.shape(var_0)
        var_2 = var_1.__getitem__(-1)
        var_3 = paddle.tensor.creation.linspace(-1, 1, var_2, dtype='float32')
        var_4 = paddle.tensor.attribute.shape(var_0)
        var_5 = var_4.__getitem__(-2)
        var_6 = paddle.tensor.creation.linspace(-1, 1, var_5, dtype='float32')
        out = paddle.tensor.creation.meshgrid([var_6, var_3])
        var_7 = out[0]
        var_8 = out[1]
        var_9 = paddle.tensor.manipulation.unsqueeze(var_8, [0, 1])
        var_10 = paddle.tensor.manipulation.unsqueeze(var_7, [0, 1])
        var_11 = paddle.tensor.attribute.shape(var_0)
        var_12 = var_11.__getitem__(0)
        var_13 = paddle.tensor.manipulation.expand(var_10, shape=[var_12, 1, -1, -1])
        var_14 = paddle.tensor.attribute.shape(var_0)
        var_15 = var_14.__getitem__(0)
        var_16 = paddle.tensor.manipulation.expand(var_9, shape=[var_15, 1, -1, -1])
        var_17 = paddle.tensor.manipulation.concat([var_16, var_13], axis=1)
        var_18 = paddle.tensor.manipulation.concat([var_0, var_17], axis=1)
        return var_18, var_3, var_6, var_13, var_16, var_17


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 256, 25, 38], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 256, 25, 38]).astype('float32'),
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