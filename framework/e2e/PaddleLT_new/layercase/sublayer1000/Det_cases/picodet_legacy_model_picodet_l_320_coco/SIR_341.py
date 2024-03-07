# api:paddle.tensor.manipulation.split||api:paddle.tensor.manipulation.split||api:paddle.tensor.math.maximum||api:paddle.tensor.math.maximum||api:paddle.tensor.math.minimum||api:paddle.tensor.math.minimum||method:__sub__||method:clip||method:__sub__||method:clip||method:__mul__||method:__sub__||method:__sub__||method:__mul__||method:__sub__||method:__sub__||method:__mul__||method:__add__||method:__sub__||method:__add__||method:__truediv__||api:paddle.tensor.math.minimum||api:paddle.tensor.math.minimum||api:paddle.tensor.math.maximum||api:paddle.tensor.math.maximum||method:__sub__||method:__sub__||method:__mul__||method:__add__||method:__sub__||method:__truediv__||method:__sub__||method:__rsub__||method:__mul__
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [6, 4], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [6, 4], dtype: paddle.float32, stop_gradient: True)
    ):
        out = paddle.tensor.manipulation.split(var_0, num_or_sections=4, axis=-1)
        var_2 = out[0]
        var_3 = out[1]
        var_4 = out[2]
        var_5 = out[3]
        out = paddle.tensor.manipulation.split(var_1, num_or_sections=4, axis=-1)
        var_6 = out[0]
        var_7 = out[1]
        var_8 = out[2]
        var_9 = out[3]
        var_10 = paddle.tensor.math.maximum(var_2, var_6)
        var_11 = paddle.tensor.math.maximum(var_3, var_7)
        var_12 = paddle.tensor.math.minimum(var_4, var_8)
        var_13 = paddle.tensor.math.minimum(var_5, var_9)
        var_14 = var_12.__sub__(var_10)
        var_15 = var_14.clip(0)
        var_16 = var_13.__sub__(var_11)
        var_17 = var_16.clip(0)
        var_18 = var_15.__mul__(var_17)
        var_19 = var_4.__sub__(var_2)
        var_20 = var_5.__sub__(var_3)
        var_21 = var_19.__mul__(var_20)
        var_22 = var_8.__sub__(var_6)
        var_23 = var_9.__sub__(var_7)
        var_24 = var_22.__mul__(var_23)
        var_25 = var_21.__add__(var_24)
        var_26 = var_25.__sub__(var_18)
        var_27 = var_26.__add__(1e-10)
        var_28 = var_18.__truediv__(var_27)
        var_29 = paddle.tensor.math.minimum(var_2, var_6)
        var_30 = paddle.tensor.math.minimum(var_3, var_7)
        var_31 = paddle.tensor.math.maximum(var_4, var_8)
        var_32 = paddle.tensor.math.maximum(var_5, var_9)
        var_33 = var_31.__sub__(var_29)
        var_34 = var_32.__sub__(var_30)
        var_35 = var_33.__mul__(var_34)
        var_36 = var_35.__add__(1e-10)
        var_37 = var_36.__sub__(var_27)
        var_38 = var_37.__truediv__(var_36)
        var_39 = var_28.__sub__(var_38)
        var_40 = var_39.__rsub__(1)
        var_41 = var_40.__mul__(2.0)
        return var_41


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[6, 4], dtype=paddle.float32),
        paddle.rand(shape=[6, 4], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[6, 4]).astype('float32'),
        np.random.random(size=[6, 4]).astype('float32'),
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