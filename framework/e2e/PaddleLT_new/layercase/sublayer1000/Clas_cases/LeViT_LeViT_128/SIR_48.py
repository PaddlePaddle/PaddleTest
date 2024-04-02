# api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.split||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.transpose||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||api:paddle.tensor.manipulation.concat||api:paddle.tensor.linalg.transpose||method:reshape
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[8, 49],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [22, 49, 512], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [49, 49], dtype: paddle.int64, stop_gradient: True)
    ):
        var_2 = paddle.tensor.manipulation.reshape(var_0, [22, 49, 8, 64])
        out = paddle.tensor.manipulation.split(var_2, [16, 16, 32], axis=3)
        var_3 = out[0]
        var_4 = out[1]
        var_5 = out[2]
        var_6 = paddle.tensor.linalg.transpose(var_3, perm=[0, 2, 1, 3])
        var_7 = paddle.tensor.linalg.transpose(var_4, perm=[0, 2, 1, 3])
        var_8 = paddle.tensor.linalg.transpose(var_5, perm=[0, 2, 1, 3])
        var_9 = paddle.tensor.linalg.transpose(var_7, perm=[0, 1, 3, 2])
        var_10 = paddle.tensor.linalg.transpose(self.parameter_0, (1, 0,))
        var_11 = var_1.__getitem__(0)
        var_12 = paddle.tensor.manipulation.gather(var_10, var_11)
        var_13 = var_1.__getitem__(1)
        var_14 = paddle.tensor.manipulation.gather(var_10, var_13)
        var_15 = var_1.__getitem__(2)
        var_16 = paddle.tensor.manipulation.gather(var_10, var_15)
        var_17 = var_1.__getitem__(3)
        var_18 = paddle.tensor.manipulation.gather(var_10, var_17)
        var_19 = var_1.__getitem__(4)
        var_20 = paddle.tensor.manipulation.gather(var_10, var_19)
        var_21 = var_1.__getitem__(5)
        var_22 = paddle.tensor.manipulation.gather(var_10, var_21)
        var_23 = var_1.__getitem__(6)
        var_24 = paddle.tensor.manipulation.gather(var_10, var_23)
        var_25 = var_1.__getitem__(7)
        var_26 = paddle.tensor.manipulation.gather(var_10, var_25)
        var_27 = var_1.__getitem__(8)
        var_28 = paddle.tensor.manipulation.gather(var_10, var_27)
        var_29 = var_1.__getitem__(9)
        var_30 = paddle.tensor.manipulation.gather(var_10, var_29)
        var_31 = var_1.__getitem__(10)
        var_32 = paddle.tensor.manipulation.gather(var_10, var_31)
        var_33 = var_1.__getitem__(11)
        var_34 = paddle.tensor.manipulation.gather(var_10, var_33)
        var_35 = var_1.__getitem__(12)
        var_36 = paddle.tensor.manipulation.gather(var_10, var_35)
        var_37 = var_1.__getitem__(13)
        var_38 = paddle.tensor.manipulation.gather(var_10, var_37)
        var_39 = var_1.__getitem__(14)
        var_40 = paddle.tensor.manipulation.gather(var_10, var_39)
        var_41 = var_1.__getitem__(15)
        var_42 = paddle.tensor.manipulation.gather(var_10, var_41)
        var_43 = var_1.__getitem__(16)
        var_44 = paddle.tensor.manipulation.gather(var_10, var_43)
        var_45 = var_1.__getitem__(17)
        var_46 = paddle.tensor.manipulation.gather(var_10, var_45)
        var_47 = var_1.__getitem__(18)
        var_48 = paddle.tensor.manipulation.gather(var_10, var_47)
        var_49 = var_1.__getitem__(19)
        var_50 = paddle.tensor.manipulation.gather(var_10, var_49)
        var_51 = var_1.__getitem__(20)
        var_52 = paddle.tensor.manipulation.gather(var_10, var_51)
        var_53 = var_1.__getitem__(21)
        var_54 = paddle.tensor.manipulation.gather(var_10, var_53)
        var_55 = var_1.__getitem__(22)
        var_56 = paddle.tensor.manipulation.gather(var_10, var_55)
        var_57 = var_1.__getitem__(23)
        var_58 = paddle.tensor.manipulation.gather(var_10, var_57)
        var_59 = var_1.__getitem__(24)
        var_60 = paddle.tensor.manipulation.gather(var_10, var_59)
        var_61 = var_1.__getitem__(25)
        var_62 = paddle.tensor.manipulation.gather(var_10, var_61)
        var_63 = var_1.__getitem__(26)
        var_64 = paddle.tensor.manipulation.gather(var_10, var_63)
        var_65 = var_1.__getitem__(27)
        var_66 = paddle.tensor.manipulation.gather(var_10, var_65)
        var_67 = var_1.__getitem__(28)
        var_68 = paddle.tensor.manipulation.gather(var_10, var_67)
        var_69 = var_1.__getitem__(29)
        var_70 = paddle.tensor.manipulation.gather(var_10, var_69)
        var_71 = var_1.__getitem__(30)
        var_72 = paddle.tensor.manipulation.gather(var_10, var_71)
        var_73 = var_1.__getitem__(31)
        var_74 = paddle.tensor.manipulation.gather(var_10, var_73)
        var_75 = var_1.__getitem__(32)
        var_76 = paddle.tensor.manipulation.gather(var_10, var_75)
        var_77 = var_1.__getitem__(33)
        var_78 = paddle.tensor.manipulation.gather(var_10, var_77)
        var_79 = var_1.__getitem__(34)
        var_80 = paddle.tensor.manipulation.gather(var_10, var_79)
        var_81 = var_1.__getitem__(35)
        var_82 = paddle.tensor.manipulation.gather(var_10, var_81)
        var_83 = var_1.__getitem__(36)
        var_84 = paddle.tensor.manipulation.gather(var_10, var_83)
        var_85 = var_1.__getitem__(37)
        var_86 = paddle.tensor.manipulation.gather(var_10, var_85)
        var_87 = var_1.__getitem__(38)
        var_88 = paddle.tensor.manipulation.gather(var_10, var_87)
        var_89 = var_1.__getitem__(39)
        var_90 = paddle.tensor.manipulation.gather(var_10, var_89)
        var_91 = var_1.__getitem__(40)
        var_92 = paddle.tensor.manipulation.gather(var_10, var_91)
        var_93 = var_1.__getitem__(41)
        var_94 = paddle.tensor.manipulation.gather(var_10, var_93)
        var_95 = var_1.__getitem__(42)
        var_96 = paddle.tensor.manipulation.gather(var_10, var_95)
        var_97 = var_1.__getitem__(43)
        var_98 = paddle.tensor.manipulation.gather(var_10, var_97)
        var_99 = var_1.__getitem__(44)
        var_100 = paddle.tensor.manipulation.gather(var_10, var_99)
        var_101 = var_1.__getitem__(45)
        var_102 = paddle.tensor.manipulation.gather(var_10, var_101)
        var_103 = var_1.__getitem__(46)
        var_104 = paddle.tensor.manipulation.gather(var_10, var_103)
        var_105 = var_1.__getitem__(47)
        var_106 = paddle.tensor.manipulation.gather(var_10, var_105)
        var_107 = var_1.__getitem__(48)
        var_108 = paddle.tensor.manipulation.gather(var_10, var_107)
        var_109 = paddle.tensor.manipulation.concat([var_12, var_14, var_16, var_18, var_20, var_22, var_24, var_26, var_28, var_30, var_32, var_34, var_36, var_38, var_40, var_42, var_44, var_46, var_48, var_50, var_52, var_54, var_56, var_58, var_60, var_62, var_64, var_66, var_68, var_70, var_72, var_74, var_76, var_78, var_80, var_82, var_84, var_86, var_88, var_90, var_92, var_94, var_96, var_98, var_100, var_102, var_104, var_106, var_108])
        var_110 = paddle.tensor.linalg.transpose(var_109, (1, 0,))
        var_111 = var_110.reshape((0, 49, 49,))
        return var_6, var_9, var_111, var_8


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[22, 49, 512], dtype=paddle.float32),
        paddle.randint(low=0, high=10, shape=[49, 49], dtype=paddle.int64),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[22, 49, 512]).astype('float32'),
        np.random.randint(low=0, high=10, size=[49, 49], dtype='int64'),
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