# api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:__getitem__||api:paddle.nn.functional.norm.layer_norm||method:reshape||api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:__getitem__||method:__getitem__
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[768],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[768],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [43, 49, 768], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = paddle.tensor.attribute.shape(var_0)
        var_2 = var_1.__getitem__(0)
        var_3 = var_1.__getitem__(1)
        var_4 = var_1.__getitem__(2)
        var_5 = paddle.nn.functional.norm.layer_norm(var_0, normalized_shape=[768], weight=self.parameter_1, bias=self.parameter_0, epsilon=1e-05)
        var_6 = var_5.reshape([var_2, 7, 7, var_4])
        var_7 = paddle.tensor.attribute.shape(var_6)
        var_8 = var_7.__getitem__(0)
        var_9 = var_7.__getitem__(1)
        var_10 = var_7.__getitem__(2)
        var_11 = var_7.__getitem__(3)
        return var_6, var_4, var_9, var_10, var_2


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[43, 49, 768], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[43, 49, 768]).astype('float32'),
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