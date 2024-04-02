# method:clone||method:unbind||method:__rmul__||method:__sub__||method:__rmul__||method:__sub__||method:__rmul__||method:__add__||method:__rmul__||method:__add__||api:paddle.tensor.manipulation.stack||method:unsqueeze||method:unsqueeze||method:__mul__||method:unsqueeze||method:tile||method:clone
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[300, 4],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[300, 256],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 4], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1 = self.parameter_0.clone()
        out = var_1.unbind(-1)
        var_2 = out[0]
        var_3 = out[1]
        var_4 = out[2]
        var_5 = out[3]
        var_6 = var_4.__rmul__(0.5)
        var_7 = var_2.__sub__(var_6)
        var_8 = var_5.__rmul__(0.5)
        var_9 = var_3.__sub__(var_8)
        var_10 = var_4.__rmul__(0.5)
        var_11 = var_2.__add__(var_10)
        var_12 = var_5.__rmul__(0.5)
        var_13 = var_3.__add__(var_12)
        var_14 = paddle.tensor.manipulation.stack([var_7, var_9, var_11, var_13], axis=-1)
        var_15 = var_14.unsqueeze(0)
        var_16 = var_0.unsqueeze(-2)
        var_17 = var_15.__mul__(var_16)
        var_18 = self.parameter_1.unsqueeze(0)
        var_19 = var_18.tile([1, 1, 1])
        var_20 = var_19.clone()
        return var_20, var_17


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 4], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 4]).astype('float32'),
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