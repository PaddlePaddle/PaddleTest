# api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.split||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.transpose||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||api:paddle.tensor.manipulation.concat||api:paddle.tensor.linalg.transpose||method:reshape
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[12, 16],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [22, 16, 768], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [16, 16], dtype: paddle.int64, stop_gradient: True)
    ):
        var_2 = paddle.tensor.manipulation.reshape(var_0, [22, 16, 12, 64])
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
        var_43 = paddle.tensor.manipulation.concat([var_12, var_14, var_16, var_18, var_20, var_22, var_24, var_26, var_28, var_30, var_32, var_34, var_36, var_38, var_40, var_42])
        var_44 = paddle.tensor.linalg.transpose(var_43, (1, 0,))
        var_45 = var_44.reshape((0, 16, 16,))
        return var_6, var_9, var_45, var_8


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[22, 16, 768], dtype=paddle.float32),
        paddle.randint(low=0, high=10, shape=[16, 16], dtype=paddle.int64),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[22, 16, 768]).astype('float32'),
        np.random.randint(low=0, high=10, size=[16, 16], dtype='int64'),
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