# api:paddle.tensor.manipulation.split||api:paddle.tensor.manipulation.split||method:__add__||method:__truediv__||method:__add__||method:__truediv__||method:__sub__||method:__sub__||method:__add__||method:__truediv__||method:__add__||method:__truediv__||method:__sub__||method:__sub__||api:paddle.tensor.math.maximum||api:paddle.tensor.math.maximum||api:paddle.tensor.math.maximum||api:paddle.tensor.math.maximum||api:paddle.tensor.math.minimum||api:paddle.tensor.math.minimum||api:paddle.tensor.math.minimum||api:paddle.tensor.math.minimum||api:paddle.tensor.math.maximum||api:paddle.tensor.math.maximum||method:__sub__||method:__sub__||method:__mul__||api:paddle.tensor.logic.greater_than||method:__mul__||api:paddle.tensor.logic.greater_than||method:__mul__||method:__sub__||method:__sub__||method:__mul__||method:__sub__||method:__sub__||method:__mul__||method:__add__||method:__sub__||method:__add__||method:__truediv__||method:__sub__||method:__sub__||method:__mul__||method:__sub__||method:__sub__||method:__mul__||method:__add__||method:__sub__||method:__sub__||method:__mul__||method:__sub__||method:__sub__||method:__mul__||method:__add__||method:__add__||method:__add__||method:__truediv__||method:__truediv__||method:__truediv__||api:paddle.tensor.ops.atan||api:paddle.tensor.ops.atan||method:__sub__||method:__rmul__||method:__mul__||method:__rsub__||method:__add__||method:__add__||method:__truediv__||method:__mul__||method:__rsub__||method:__add__||method:__add__||method:__mul__||api:paddle.tensor.stat.mean||method:__mul__
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [8], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [8], dtype: paddle.float32, stop_gradient: True)
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
        var_10 = var_2.__add__(var_4)
        var_11 = var_10.__truediv__(2)
        var_12 = var_3.__add__(var_5)
        var_13 = var_12.__truediv__(2)
        var_14 = var_4.__sub__(var_2)
        var_15 = var_5.__sub__(var_3)
        var_16 = var_6.__add__(var_8)
        var_17 = var_16.__truediv__(2)
        var_18 = var_7.__add__(var_9)
        var_19 = var_18.__truediv__(2)
        var_20 = var_8.__sub__(var_6)
        var_21 = var_9.__sub__(var_7)
        var_22 = paddle.tensor.math.maximum(var_2, var_4)
        var_23 = paddle.tensor.math.maximum(var_3, var_5)
        var_24 = paddle.tensor.math.maximum(var_2, var_6)
        var_25 = paddle.tensor.math.maximum(var_3, var_7)
        var_26 = paddle.tensor.math.minimum(var_22, var_8)
        var_27 = paddle.tensor.math.minimum(var_23, var_9)
        var_28 = paddle.tensor.math.minimum(var_2, var_6)
        var_29 = paddle.tensor.math.minimum(var_3, var_7)
        var_30 = paddle.tensor.math.maximum(var_22, var_8)
        var_31 = paddle.tensor.math.maximum(var_23, var_9)
        var_32 = var_26.__sub__(var_24)
        var_33 = var_27.__sub__(var_25)
        var_34 = var_32.__mul__(var_33)
        var_35 = paddle.tensor.logic.greater_than(var_26, var_24)
        var_36 = var_34.__mul__(var_35)
        var_37 = paddle.tensor.logic.greater_than(var_27, var_25)
        var_38 = var_36.__mul__(var_37)
        var_39 = var_22.__sub__(var_2)
        var_40 = var_23.__sub__(var_3)
        var_41 = var_39.__mul__(var_40)
        var_42 = var_8.__sub__(var_6)
        var_43 = var_9.__sub__(var_7)
        var_44 = var_42.__mul__(var_43)
        var_45 = var_41.__add__(var_44)
        var_46 = var_45.__sub__(var_38)
        var_47 = var_46.__add__(1e-10)
        var_48 = var_38.__truediv__(var_47)
        var_49 = var_11.__sub__(var_17)
        var_50 = var_11.__sub__(var_17)
        var_51 = var_49.__mul__(var_50)
        var_52 = var_13.__sub__(var_19)
        var_53 = var_13.__sub__(var_19)
        var_54 = var_52.__mul__(var_53)
        var_55 = var_51.__add__(var_54)
        var_56 = var_30.__sub__(var_28)
        var_57 = var_30.__sub__(var_28)
        var_58 = var_56.__mul__(var_57)
        var_59 = var_31.__sub__(var_29)
        var_60 = var_31.__sub__(var_29)
        var_61 = var_59.__mul__(var_60)
        var_62 = var_58.__add__(var_61)
        var_63 = var_55.__add__(1e-10)
        var_64 = var_62.__add__(1e-10)
        var_65 = var_63.__truediv__(var_64)
        var_66 = var_20.__truediv__(var_21)
        var_67 = var_14.__truediv__(var_15)
        var_68 = paddle.tensor.ops.atan(var_66)
        var_69 = paddle.tensor.ops.atan(var_67)
        var_70 = var_68.__sub__(var_69)
        var_71 = var_70.__rmul__(0.4052847345693511)
        var_72 = var_71.__mul__(var_70)
        var_73 = var_48.__rsub__(1)
        var_74 = var_73.__add__(var_72)
        var_75 = var_74.__add__(1e-10)
        var_76 = var_72.__truediv__(var_75)
        var_77 = var_76.__mul__(var_72)
        var_78 = var_48.__rsub__(1)
        var_79 = var_78.__add__(var_77)
        var_80 = var_79.__add__(var_65)
        var_81 = var_80.__mul__(1.0)
        var_82 = paddle.tensor.stat.mean(var_81)
        var_83 = var_82.__mul__(10.0)
        return var_83


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[8], dtype=paddle.float32),
        paddle.rand(shape=[8], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[8]).astype('float32'),
        np.random.random(size=[8]).astype('float32'),
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