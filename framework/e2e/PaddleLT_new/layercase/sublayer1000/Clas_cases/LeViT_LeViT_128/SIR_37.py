# api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.transpose||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||api:paddle.tensor.manipulation.concat||api:paddle.tensor.linalg.transpose||method:reshape||api:paddle.tensor.linalg.transpose
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[8, 196],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [22, 49, 128], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [22, 8, 196, 16], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [49, 196], dtype: paddle.int64, stop_gradient: True)
    ):
        var_3 = paddle.tensor.manipulation.reshape(var_0, [22, 49, 8, 16])
        var_4 = paddle.tensor.linalg.transpose(var_3, perm=[0, 2, 1, 3])
        var_5 = paddle.tensor.linalg.transpose(self.parameter_0, (1, 0,))
        var_6 = var_2.__getitem__(0)
        var_7 = paddle.tensor.manipulation.gather(var_5, var_6)
        var_8 = var_2.__getitem__(1)
        var_9 = paddle.tensor.manipulation.gather(var_5, var_8)
        var_10 = var_2.__getitem__(2)
        var_11 = paddle.tensor.manipulation.gather(var_5, var_10)
        var_12 = var_2.__getitem__(3)
        var_13 = paddle.tensor.manipulation.gather(var_5, var_12)
        var_14 = var_2.__getitem__(4)
        var_15 = paddle.tensor.manipulation.gather(var_5, var_14)
        var_16 = var_2.__getitem__(5)
        var_17 = paddle.tensor.manipulation.gather(var_5, var_16)
        var_18 = var_2.__getitem__(6)
        var_19 = paddle.tensor.manipulation.gather(var_5, var_18)
        var_20 = var_2.__getitem__(7)
        var_21 = paddle.tensor.manipulation.gather(var_5, var_20)
        var_22 = var_2.__getitem__(8)
        var_23 = paddle.tensor.manipulation.gather(var_5, var_22)
        var_24 = var_2.__getitem__(9)
        var_25 = paddle.tensor.manipulation.gather(var_5, var_24)
        var_26 = var_2.__getitem__(10)
        var_27 = paddle.tensor.manipulation.gather(var_5, var_26)
        var_28 = var_2.__getitem__(11)
        var_29 = paddle.tensor.manipulation.gather(var_5, var_28)
        var_30 = var_2.__getitem__(12)
        var_31 = paddle.tensor.manipulation.gather(var_5, var_30)
        var_32 = var_2.__getitem__(13)
        var_33 = paddle.tensor.manipulation.gather(var_5, var_32)
        var_34 = var_2.__getitem__(14)
        var_35 = paddle.tensor.manipulation.gather(var_5, var_34)
        var_36 = var_2.__getitem__(15)
        var_37 = paddle.tensor.manipulation.gather(var_5, var_36)
        var_38 = var_2.__getitem__(16)
        var_39 = paddle.tensor.manipulation.gather(var_5, var_38)
        var_40 = var_2.__getitem__(17)
        var_41 = paddle.tensor.manipulation.gather(var_5, var_40)
        var_42 = var_2.__getitem__(18)
        var_43 = paddle.tensor.manipulation.gather(var_5, var_42)
        var_44 = var_2.__getitem__(19)
        var_45 = paddle.tensor.manipulation.gather(var_5, var_44)
        var_46 = var_2.__getitem__(20)
        var_47 = paddle.tensor.manipulation.gather(var_5, var_46)
        var_48 = var_2.__getitem__(21)
        var_49 = paddle.tensor.manipulation.gather(var_5, var_48)
        var_50 = var_2.__getitem__(22)
        var_51 = paddle.tensor.manipulation.gather(var_5, var_50)
        var_52 = var_2.__getitem__(23)
        var_53 = paddle.tensor.manipulation.gather(var_5, var_52)
        var_54 = var_2.__getitem__(24)
        var_55 = paddle.tensor.manipulation.gather(var_5, var_54)
        var_56 = var_2.__getitem__(25)
        var_57 = paddle.tensor.manipulation.gather(var_5, var_56)
        var_58 = var_2.__getitem__(26)
        var_59 = paddle.tensor.manipulation.gather(var_5, var_58)
        var_60 = var_2.__getitem__(27)
        var_61 = paddle.tensor.manipulation.gather(var_5, var_60)
        var_62 = var_2.__getitem__(28)
        var_63 = paddle.tensor.manipulation.gather(var_5, var_62)
        var_64 = var_2.__getitem__(29)
        var_65 = paddle.tensor.manipulation.gather(var_5, var_64)
        var_66 = var_2.__getitem__(30)
        var_67 = paddle.tensor.manipulation.gather(var_5, var_66)
        var_68 = var_2.__getitem__(31)
        var_69 = paddle.tensor.manipulation.gather(var_5, var_68)
        var_70 = var_2.__getitem__(32)
        var_71 = paddle.tensor.manipulation.gather(var_5, var_70)
        var_72 = var_2.__getitem__(33)
        var_73 = paddle.tensor.manipulation.gather(var_5, var_72)
        var_74 = var_2.__getitem__(34)
        var_75 = paddle.tensor.manipulation.gather(var_5, var_74)
        var_76 = var_2.__getitem__(35)
        var_77 = paddle.tensor.manipulation.gather(var_5, var_76)
        var_78 = var_2.__getitem__(36)
        var_79 = paddle.tensor.manipulation.gather(var_5, var_78)
        var_80 = var_2.__getitem__(37)
        var_81 = paddle.tensor.manipulation.gather(var_5, var_80)
        var_82 = var_2.__getitem__(38)
        var_83 = paddle.tensor.manipulation.gather(var_5, var_82)
        var_84 = var_2.__getitem__(39)
        var_85 = paddle.tensor.manipulation.gather(var_5, var_84)
        var_86 = var_2.__getitem__(40)
        var_87 = paddle.tensor.manipulation.gather(var_5, var_86)
        var_88 = var_2.__getitem__(41)
        var_89 = paddle.tensor.manipulation.gather(var_5, var_88)
        var_90 = var_2.__getitem__(42)
        var_91 = paddle.tensor.manipulation.gather(var_5, var_90)
        var_92 = var_2.__getitem__(43)
        var_93 = paddle.tensor.manipulation.gather(var_5, var_92)
        var_94 = var_2.__getitem__(44)
        var_95 = paddle.tensor.manipulation.gather(var_5, var_94)
        var_96 = var_2.__getitem__(45)
        var_97 = paddle.tensor.manipulation.gather(var_5, var_96)
        var_98 = var_2.__getitem__(46)
        var_99 = paddle.tensor.manipulation.gather(var_5, var_98)
        var_100 = var_2.__getitem__(47)
        var_101 = paddle.tensor.manipulation.gather(var_5, var_100)
        var_102 = var_2.__getitem__(48)
        var_103 = paddle.tensor.manipulation.gather(var_5, var_102)
        var_104 = paddle.tensor.manipulation.concat([var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23, var_25, var_27, var_29, var_31, var_33, var_35, var_37, var_39, var_41, var_43, var_45, var_47, var_49, var_51, var_53, var_55, var_57, var_59, var_61, var_63, var_65, var_67, var_69, var_71, var_73, var_75, var_77, var_79, var_81, var_83, var_85, var_87, var_89, var_91, var_93, var_95, var_97, var_99, var_101, var_103])
        var_105 = paddle.tensor.linalg.transpose(var_104, (1, 0,))
        var_106 = var_105.reshape((0, 49, 196,))
        var_107 = paddle.tensor.linalg.transpose(var_1, perm=[0, 1, 3, 2])
        return var_4, var_107, var_106


def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[22, 49, 128], dtype=paddle.float32),
        paddle.rand(shape=[22, 8, 196, 16], dtype=paddle.float32),
        paddle.randint(low=0, high=10, shape=[49, 196], dtype=paddle.int64),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[22, 49, 128]).astype('float32'),
        np.random.random(size=[22, 8, 196, 16]).astype('float32'),
        np.random.randint(low=0, high=10, size=[49, 196], dtype='int64'),
    )
    return inputs


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = create_tensor_inputs()
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