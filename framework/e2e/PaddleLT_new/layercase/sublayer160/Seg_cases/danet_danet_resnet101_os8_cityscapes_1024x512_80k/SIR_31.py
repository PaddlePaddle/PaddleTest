# api:paddle.tensor.attribute.shape||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.bmm||api:paddle.tensor.math.max||method:tile||method:__sub__||api:paddle.nn.functional.activation.softmax||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.bmm||method:__getitem__||method:__getitem__||api:paddle.tensor.manipulation.reshape||method:__mul__||method:__add__
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[1],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 512, 64, 128], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = paddle.tensor.attribute.shape(var_0)
        var_2 = paddle.tensor.manipulation.reshape(var_0, (0, 512, -1,))
        var_3 = paddle.tensor.manipulation.reshape(var_0, (0, 512, -1,))
        var_4 = paddle.tensor.linalg.transpose(var_3, (0, 2, 1,))
        var_5 = paddle.tensor.linalg.bmm(var_2, var_4)
        var_6 = paddle.tensor.math.max(var_5, axis=-1, keepdim=True)
        var_7 = var_6.tile([1, 1, 512])
        var_8 = var_7.__sub__(var_5)
        var_9 = paddle.nn.functional.activation.softmax(var_8, axis=-1)
        var_10 = paddle.tensor.manipulation.reshape(var_0, (0, 512, -1,))
        var_11 = paddle.tensor.linalg.bmm(var_9, var_10)
        var_12 = var_1.__getitem__(2)
        var_13 = var_1.__getitem__(3)
        var_14 = paddle.tensor.manipulation.reshape(var_11, (0, 512, var_12, var_13,))
        var_15 = self.parameter_0.__mul__(var_14)
        var_16 = var_15.__add__(var_0)
        return var_16


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 512, 64, 128], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 512, 64, 128]).astype('float32'),
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