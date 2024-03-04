# method:split||method:unsqueeze||method:__sub__||method:__sub__||method:split||method:unsqueeze||method:__sub__||method:__mul__||api:paddle.tensor.math.sum||method:__mul__||api:paddle.tensor.math.sum||method:__mul__||api:paddle.tensor.math.sum||method:sqrt||method:__mul__||api:paddle.tensor.math.sum||method:sqrt||api:paddle.tensor.math.min||method:pow||method:pow||method:__mul__||method:__add__||method:__truediv__||method:pow||method:pow||method:__mul__||method:__add__||method:__truediv__||method:__add__||method:__rmul__||api:paddle.tensor.ops.exp||method:__truediv__||method:__rmul__||method:__add__||method:__truediv__
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [1, 21824, 2], dtype: paddle.float32, stop_gradient: True)
        var_1,    # (shape: [1, 6, 5], dtype: paddle.float32, stop_gradient: True)
        var_2,    # (shape: [1, 6, 4, 2], dtype: paddle.float32, stop_gradient: True)
    ):
        out = var_2.split(4, axis=2)
        var_3 = out[0]
        var_4 = out[1]
        var_5 = out[2]
        var_6 = out[3]
        var_7 = var_0.unsqueeze(0)
        var_8 = var_4.__sub__(var_3)
        var_9 = var_6.__sub__(var_3)
        out = var_1.split([2, 2, 1], axis=-1)
        var_10 = out[0]
        var_11 = out[1]
        var_12 = out[2]
        var_13 = var_10.unsqueeze(2)
        var_14 = var_7.__sub__(var_13)
        var_15 = var_14.__mul__(var_8)
        var_16 = paddle.tensor.math.sum(var_15, axis=-1)
        var_17 = var_14.__mul__(var_9)
        var_18 = paddle.tensor.math.sum(var_17, axis=-1)
        var_19 = var_8.__mul__(var_8)
        var_20 = paddle.tensor.math.sum(var_19, axis=-1)
        var_21 = var_20.sqrt()
        var_22 = var_9.__mul__(var_9)
        var_23 = paddle.tensor.math.sum(var_22, axis=-1)
        var_24 = var_23.sqrt()
        var_25 = paddle.tensor.math.min(var_11, axis=-1, keepdim=True)
        var_26 = var_16.pow(2)
        var_27 = var_21.pow(3)
        var_28 = var_27.__mul__(var_25)
        var_29 = var_28.__add__(1e-09)
        var_30 = var_26.__truediv__(var_29)
        var_31 = var_18.pow(2)
        var_32 = var_24.pow(3)
        var_33 = var_32.__mul__(var_25)
        var_34 = var_33.__add__(1e-09)
        var_35 = var_31.__truediv__(var_34)
        var_36 = var_30.__add__(var_35)
        var_37 = var_36.__rmul__(-6.0)
        var_38 = paddle.tensor.ops.exp(var_37)
        var_39 = var_25.__truediv__(12)
        var_40 = var_39.__rmul__(6.283185307179586)
        var_41 = var_40.__add__(1e-09)
        var_42 = var_38.__truediv__(var_41)
        return var_38, var_42


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 21824, 2], dtype=paddle.float32),
        paddle.rand(shape=[1, 6, 5], dtype=paddle.float32),
        paddle.rand(shape=[1, 6, 4, 2], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 21824, 2]).astype('float32'),
        np.random.random(size=[1, 6, 5]).astype('float32'),
        np.random.random(size=[1, 6, 4, 2]).astype('float32'),
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