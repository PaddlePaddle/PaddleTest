# api:paddle.tensor.manipulation.concat||method:__pow__||method:__rmul__||method:__rsub__||method:__add__||method:log||method:__neg__||method:__mul__||method:__rsub__||method:__pow__||method:__rmul__||method:__add__||method:log||method:__neg__||method:__mul__||api:paddle.tensor.manipulation.gather||api:paddle.tensor.manipulation.gather||method:__sub__||method:unsqueeze||api:paddle.tensor.manipulation.concat||method:unsqueeze||method:tile||method:flatten||api:paddle.tensor.manipulation.concat||method:__truediv__||method:__truediv__||method:unsqueeze||api:paddle.nn.functional.loss.l1_loss||method:sum
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [300, 80], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [2], dtype: paddle.int32, stop_gradient: True)
        var_2,    # (shape: [300, 4], dtype: paddle.float32, stop_gradient: False)
        var_3,    # (shape: [2, 4], dtype: paddle.float32, stop_gradient: True)
        var_4,    # (shape: [4], dtype: paddle.float32, stop_gradient: True)
        var_5,    # (shape: [2, 4], dtype: paddle.float32, stop_gradient: True)
    ):
        var_6 = paddle.tensor.manipulation.concat([var_3])
        var_7 = var_0.__pow__(2.0)
        var_8 = var_7.__rmul__(0.75)
        var_9 = var_0.__rsub__(1)
        var_10 = var_9.__add__(1e-08)
        var_11 = var_10.log()
        var_12 = var_11.__neg__()
        var_13 = var_8.__mul__(var_12)
        var_14 = var_0.__rsub__(1)
        var_15 = var_14.__pow__(2.0)
        var_16 = var_15.__rmul__(0.25)
        var_17 = var_0.__add__(1e-08)
        var_18 = var_17.log()
        var_19 = var_18.__neg__()
        var_20 = var_16.__mul__(var_19)
        var_21 = paddle.tensor.manipulation.gather(var_20, var_1, axis=1)
        var_22 = paddle.tensor.manipulation.gather(var_13, var_1, axis=1)
        var_23 = var_21.__sub__(var_22)
        var_24 = var_4.unsqueeze(0)
        var_25 = paddle.tensor.manipulation.concat([var_24])
        var_26 = var_25.unsqueeze(1)
        var_27 = var_26.tile([1, 300, 1])
        var_28 = var_27.flatten(start_axis=0, stop_axis=1)
        var_29 = paddle.tensor.manipulation.concat([var_5])
        var_30 = var_2.__truediv__(var_28)
        var_31 = var_6.__truediv__(var_29)
        var_32 = var_30.unsqueeze(-2)
        var_33 = paddle.nn.functional.loss.l1_loss(var_32, var_31, reduction='none')
        var_34 = var_33.sum(-1)
        return var_6, var_34, var_23


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[300, 80], dtype=paddle.float32),
        paddle.randint(low=0, high=10, shape=[2], dtype=paddle.int32),
        paddle.rand(shape=[300, 4], dtype=paddle.float32),
        paddle.rand(shape=[2, 4], dtype=paddle.float32),
        paddle.rand(shape=[4], dtype=paddle.float32),
        paddle.rand(shape=[2, 4], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[300, 80]).astype('float32'),
        np.random.randint(low=0, high=10, size=[2], dtype='int32'),
        np.random.random(size=[300, 4]).astype('float32'),
        np.random.random(size=[2, 4]).astype('float32'),
        np.random.random(size=[4]).astype('float32'),
        np.random.random(size=[2, 4]).astype('float32'),
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