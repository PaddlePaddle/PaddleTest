# api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:__getitem__||method:__getitem__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||api:paddle.tensor.manipulation.cast||method:reshape||method:__mul__||api:paddle.tensor.creation.full||method:__mul__||api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:__getitem__||method:__getitem__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||api:paddle.tensor.manipulation.cast||method:reshape||method:__mul__||api:paddle.tensor.creation.full||method:__mul__||api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:__getitem__||method:__getitem__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||api:paddle.tensor.manipulation.cast||method:reshape||method:__mul__||api:paddle.tensor.creation.full||method:__mul__||api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:__getitem__||method:__getitem__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||api:paddle.tensor.manipulation.cast||method:reshape||method:__mul__||api:paddle.tensor.creation.full||method:__mul__||api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:__getitem__||method:__getitem__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||api:paddle.tensor.manipulation.cast||method:reshape||method:__mul__||api:paddle.tensor.creation.full||method:__mul__||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [1, 256, 128, 128], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 256, 64, 64], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [1, 256, 32, 32], dtype: paddle.float32, stop_gradient: False)
        var_3,    # (shape: [1, 256, 16, 16], dtype: paddle.float32, stop_gradient: False)
        var_4,    # (shape: [1, 256, 8, 8], dtype: paddle.float32, stop_gradient: False)
    ):
        var_5 = paddle.tensor.attribute.shape(var_0)
        var_6 = var_5.__getitem__(0)
        var_7 = var_5.__getitem__(1)
        var_8 = var_5.__getitem__(2)
        var_9 = var_5.__getitem__(3)
        var_10 = paddle.tensor.creation.arange(end=var_9)
        var_11 = var_10.__add__(0.5)
        var_12 = var_11.__mul__(8)
        var_13 = paddle.tensor.creation.arange(end=var_8)
        var_14 = var_13.__add__(0.5)
        var_15 = var_14.__mul__(8)
        out = paddle.tensor.creation.meshgrid(var_15, var_12)
        var_16 = out[0]
        var_17 = out[1]
        var_18 = paddle.tensor.manipulation.stack([var_17, var_16], axis=-1)
        var_19 = paddle.tensor.manipulation.cast(var_18, dtype='float32')
        var_20 = var_19.reshape([1, -1, 2])
        var_21 = var_8.__mul__(var_9)
        var_22 = paddle.tensor.creation.full([1, var_21, 1], 8, dtype='float32')
        var_23 = var_8.__mul__(var_9)
        var_24 = paddle.tensor.attribute.shape(var_1)
        var_25 = var_24.__getitem__(0)
        var_26 = var_24.__getitem__(1)
        var_27 = var_24.__getitem__(2)
        var_28 = var_24.__getitem__(3)
        var_29 = paddle.tensor.creation.arange(end=var_28)
        var_30 = var_29.__add__(0.5)
        var_31 = var_30.__mul__(16)
        var_32 = paddle.tensor.creation.arange(end=var_27)
        var_33 = var_32.__add__(0.5)
        var_34 = var_33.__mul__(16)
        out = paddle.tensor.creation.meshgrid(var_34, var_31)
        var_35 = out[0]
        var_36 = out[1]
        var_37 = paddle.tensor.manipulation.stack([var_36, var_35], axis=-1)
        var_38 = paddle.tensor.manipulation.cast(var_37, dtype='float32')
        var_39 = var_38.reshape([1, -1, 2])
        var_40 = var_27.__mul__(var_28)
        var_41 = paddle.tensor.creation.full([1, var_40, 1], 16, dtype='float32')
        var_42 = var_27.__mul__(var_28)
        var_43 = paddle.tensor.attribute.shape(var_2)
        var_44 = var_43.__getitem__(0)
        var_45 = var_43.__getitem__(1)
        var_46 = var_43.__getitem__(2)
        var_47 = var_43.__getitem__(3)
        var_48 = paddle.tensor.creation.arange(end=var_47)
        var_49 = var_48.__add__(0.5)
        var_50 = var_49.__mul__(32)
        var_51 = paddle.tensor.creation.arange(end=var_46)
        var_52 = var_51.__add__(0.5)
        var_53 = var_52.__mul__(32)
        out = paddle.tensor.creation.meshgrid(var_53, var_50)
        var_54 = out[0]
        var_55 = out[1]
        var_56 = paddle.tensor.manipulation.stack([var_55, var_54], axis=-1)
        var_57 = paddle.tensor.manipulation.cast(var_56, dtype='float32')
        var_58 = var_57.reshape([1, -1, 2])
        var_59 = var_46.__mul__(var_47)
        var_60 = paddle.tensor.creation.full([1, var_59, 1], 32, dtype='float32')
        var_61 = var_46.__mul__(var_47)
        var_62 = paddle.tensor.attribute.shape(var_3)
        var_63 = var_62.__getitem__(0)
        var_64 = var_62.__getitem__(1)
        var_65 = var_62.__getitem__(2)
        var_66 = var_62.__getitem__(3)
        var_67 = paddle.tensor.creation.arange(end=var_66)
        var_68 = var_67.__add__(0.5)
        var_69 = var_68.__mul__(64)
        var_70 = paddle.tensor.creation.arange(end=var_65)
        var_71 = var_70.__add__(0.5)
        var_72 = var_71.__mul__(64)
        out = paddle.tensor.creation.meshgrid(var_72, var_69)
        var_73 = out[0]
        var_74 = out[1]
        var_75 = paddle.tensor.manipulation.stack([var_74, var_73], axis=-1)
        var_76 = paddle.tensor.manipulation.cast(var_75, dtype='float32')
        var_77 = var_76.reshape([1, -1, 2])
        var_78 = var_65.__mul__(var_66)
        var_79 = paddle.tensor.creation.full([1, var_78, 1], 64, dtype='float32')
        var_80 = var_65.__mul__(var_66)
        var_81 = paddle.tensor.attribute.shape(var_4)
        var_82 = var_81.__getitem__(0)
        var_83 = var_81.__getitem__(1)
        var_84 = var_81.__getitem__(2)
        var_85 = var_81.__getitem__(3)
        var_86 = paddle.tensor.creation.arange(end=var_85)
        var_87 = var_86.__add__(0.5)
        var_88 = var_87.__mul__(128)
        var_89 = paddle.tensor.creation.arange(end=var_84)
        var_90 = var_89.__add__(0.5)
        var_91 = var_90.__mul__(128)
        out = paddle.tensor.creation.meshgrid(var_91, var_88)
        var_92 = out[0]
        var_93 = out[1]
        var_94 = paddle.tensor.manipulation.stack([var_93, var_92], axis=-1)
        var_95 = paddle.tensor.manipulation.cast(var_94, dtype='float32')
        var_96 = var_95.reshape([1, -1, 2])
        var_97 = var_84.__mul__(var_85)
        var_98 = paddle.tensor.creation.full([1, var_97, 1], 128, dtype='float32')
        var_99 = var_84.__mul__(var_85)
        var_100 = paddle.tensor.manipulation.concat([var_20, var_39, var_58, var_77, var_96], axis=1)
        var_101 = paddle.tensor.manipulation.concat([var_22, var_41, var_60, var_79, var_98], axis=1)
        return var_100, var_101, var_23, var_42, var_61, var_80, var_99


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 256, 128, 128], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 64, 64], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 32, 32], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 16, 16], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 8, 8], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 256, 128, 128]).astype('float32'),
        np.random.random(size=[1, 256, 64, 64]).astype('float32'),
        np.random.random(size=[1, 256, 32, 32]).astype('float32'),
        np.random.random(size=[1, 256, 16, 16]).astype('float32'),
        np.random.random(size=[1, 256, 8, 8]).astype('float32'),
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