# api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.manipulation.concat||method:astype||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.split||method:__truediv__||method:__add__||api:paddle.tensor.ops.exp||method:__mul__||method:__sub__||method:__add__||api:paddle.tensor.manipulation.concat||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.manipulation.concat||method:astype||api:paddle.tensor.manipulation.concat
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [1, 1936, 80], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 484, 80], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [1, 121, 80], dtype: paddle.float32, stop_gradient: False)
        var_3,    # (shape: [1, 1936, 4], dtype: paddle.float32, stop_gradient: False)
        var_4,    # (shape: [1, 484, 4], dtype: paddle.float32, stop_gradient: False)
        var_5,    # (shape: [1, 121, 4], dtype: paddle.float32, stop_gradient: False)
        var_6,    # (shape: [1, 1936, 1], dtype: paddle.float32, stop_gradient: False)
        var_7,    # (shape: [1, 484, 1], dtype: paddle.float32, stop_gradient: False)
        var_8,    # (shape: [1, 121, 1], dtype: paddle.float32, stop_gradient: False)
    ):
        var_9 = paddle.tensor.manipulation.concat([var_0, var_1, var_2], axis=1)
        var_10 = paddle.tensor.manipulation.concat([var_3, var_4, var_5], axis=1)
        var_11 = paddle.tensor.manipulation.concat([var_6, var_7, var_8], axis=1)
        var_12 = paddle.tensor.creation.arange(44)
        var_13 = var_12.__add__(0.0)
        var_14 = var_13.__mul__(8)
        var_15 = paddle.tensor.creation.arange(44)
        var_16 = var_15.__add__(0.0)
        var_17 = var_16.__mul__(8)
        out = paddle.tensor.creation.meshgrid(var_17, var_14)
        var_18 = out[0]
        var_19 = out[1]
        var_20 = paddle.tensor.manipulation.stack([var_19, var_18], axis=-1)
        var_21 = var_20.reshape([-1, 2])
        var_22 = paddle.tensor.creation.full([1936, 1], 8, dtype='float32')
        var_23 = paddle.tensor.creation.arange(22)
        var_24 = var_23.__add__(0.0)
        var_25 = var_24.__mul__(16)
        var_26 = paddle.tensor.creation.arange(22)
        var_27 = var_26.__add__(0.0)
        var_28 = var_27.__mul__(16)
        out = paddle.tensor.creation.meshgrid(var_28, var_25)
        var_29 = out[0]
        var_30 = out[1]
        var_31 = paddle.tensor.manipulation.stack([var_30, var_29], axis=-1)
        var_32 = var_31.reshape([-1, 2])
        var_33 = paddle.tensor.creation.full([484, 1], 16, dtype='float32')
        var_34 = paddle.tensor.creation.arange(11)
        var_35 = var_34.__add__(0.0)
        var_36 = var_35.__mul__(32)
        var_37 = paddle.tensor.creation.arange(11)
        var_38 = var_37.__add__(0.0)
        var_39 = var_38.__mul__(32)
        out = paddle.tensor.creation.meshgrid(var_39, var_36)
        var_40 = out[0]
        var_41 = out[1]
        var_42 = paddle.tensor.manipulation.stack([var_41, var_40], axis=-1)
        var_43 = var_42.reshape([-1, 2])
        var_44 = paddle.tensor.creation.full([121, 1], 32, dtype='float32')
        var_45 = paddle.tensor.manipulation.concat([var_21, var_32, var_43])
        var_46 = var_45.astype('float32')
        var_47 = paddle.tensor.manipulation.concat([var_22, var_33, var_44])
        out = paddle.tensor.manipulation.split(var_10, 2, axis=-1)
        var_48 = out[0]
        var_49 = out[1]
        var_50 = var_46.__truediv__(var_47)
        var_51 = var_48.__add__(var_50)
        var_52 = paddle.tensor.ops.exp(var_49)
        var_53 = var_52.__mul__(0.5)
        var_54 = var_51.__sub__(var_53)
        var_55 = var_51.__add__(var_53)
        var_56 = paddle.tensor.manipulation.concat([var_54, var_55], axis=-1)
        var_57 = paddle.tensor.creation.arange(44)
        var_58 = var_57.__add__(0.5)
        var_59 = var_58.__mul__(8)
        var_60 = paddle.tensor.creation.arange(44)
        var_61 = var_60.__add__(0.5)
        var_62 = var_61.__mul__(8)
        out = paddle.tensor.creation.meshgrid(var_62, var_59)
        var_63 = out[0]
        var_64 = out[1]
        var_65 = paddle.tensor.manipulation.stack([var_64, var_63], axis=-1)
        var_66 = var_65.reshape([-1, 2])
        var_67 = paddle.tensor.creation.full([1936, 1], 8, dtype='float32')
        var_68 = paddle.tensor.creation.arange(22)
        var_69 = var_68.__add__(0.5)
        var_70 = var_69.__mul__(16)
        var_71 = paddle.tensor.creation.arange(22)
        var_72 = var_71.__add__(0.5)
        var_73 = var_72.__mul__(16)
        out = paddle.tensor.creation.meshgrid(var_73, var_70)
        var_74 = out[0]
        var_75 = out[1]
        var_76 = paddle.tensor.manipulation.stack([var_75, var_74], axis=-1)
        var_77 = var_76.reshape([-1, 2])
        var_78 = paddle.tensor.creation.full([484, 1], 16, dtype='float32')
        var_79 = paddle.tensor.creation.arange(11)
        var_80 = var_79.__add__(0.5)
        var_81 = var_80.__mul__(32)
        var_82 = paddle.tensor.creation.arange(11)
        var_83 = var_82.__add__(0.5)
        var_84 = var_83.__mul__(32)
        out = paddle.tensor.creation.meshgrid(var_84, var_81)
        var_85 = out[0]
        var_86 = out[1]
        var_87 = paddle.tensor.manipulation.stack([var_86, var_85], axis=-1)
        var_88 = var_87.reshape([-1, 2])
        var_89 = paddle.tensor.creation.full([121, 1], 32, dtype='float32')
        var_90 = paddle.tensor.manipulation.concat([var_66, var_77, var_88])
        var_91 = var_90.astype('float32')
        var_92 = paddle.tensor.manipulation.concat([var_67, var_78, var_89])
        return var_9, var_56, var_11, var_91, var_92, var_66, var_77, var_88


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 1936, 80], dtype=paddle.float32),
        paddle.rand(shape=[1, 484, 80], dtype=paddle.float32),
        paddle.rand(shape=[1, 121, 80], dtype=paddle.float32),
        paddle.rand(shape=[1, 1936, 4], dtype=paddle.float32),
        paddle.rand(shape=[1, 484, 4], dtype=paddle.float32),
        paddle.rand(shape=[1, 121, 4], dtype=paddle.float32),
        paddle.rand(shape=[1, 1936, 1], dtype=paddle.float32),
        paddle.rand(shape=[1, 484, 1], dtype=paddle.float32),
        paddle.rand(shape=[1, 121, 1], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 1936, 80]).astype('float32'),
        np.random.random(size=[1, 484, 80]).astype('float32'),
        np.random.random(size=[1, 121, 80]).astype('float32'),
        np.random.random(size=[1, 1936, 4]).astype('float32'),
        np.random.random(size=[1, 484, 4]).astype('float32'),
        np.random.random(size=[1, 121, 4]).astype('float32'),
        np.random.random(size=[1, 1936, 1]).astype('float32'),
        np.random.random(size=[1, 484, 1]).astype('float32'),
        np.random.random(size=[1, 121, 1]).astype('float32'),
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