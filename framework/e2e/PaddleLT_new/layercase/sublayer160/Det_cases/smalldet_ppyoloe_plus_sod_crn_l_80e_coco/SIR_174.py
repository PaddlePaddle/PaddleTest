# method:argmax||method:__mul__||method:__add__||method:flatten||method:flatten||api:paddle.tensor.manipulation.gather||method:reshape||method:__gt__||api:paddle.tensor.creation.full_like||api:paddle.tensor.search.where||method:reshape||method:flatten||api:paddle.tensor.manipulation.gather||method:reshape||api:paddle.nn.functional.input.one_hot||api:paddle.tensor.creation.to_tensor||api:paddle.tensor.search.index_select||method:__mul__||method:max||method:__mul__||method:max||method:__add__||method:__truediv__||method:__mul__||method:max||method:unsqueeze||method:__mul__
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [1, 1, 7581], dtype: paddle.float32, stop_gradient: True)
        var_1,    # (shape: [1, 1], dtype: paddle.int32, stop_gradient: True)
        var_2,    # (shape: [1, 1, 1], dtype: paddle.int32, stop_gradient: True)
        var_3,    # (shape: [1, 7581], dtype: paddle.float32, stop_gradient: True)
        var_4,    # (shape: [1, 1, 4], dtype: paddle.float32, stop_gradient: True)
        var_5,    # (shape: [1, 1, 7581], dtype: paddle.float32, stop_gradient: True)
        var_6,    # (shape: [1, 1, 7581], dtype: paddle.float32, stop_gradient: True)
    ):
        var_7 = var_0.argmax(axis=-2)
        var_8 = var_1.__mul__(1)
        var_9 = var_7.__add__(var_8)
        var_10 = var_2.flatten()
        var_11 = var_9.flatten()
        var_12 = paddle.tensor.manipulation.gather(var_10, var_11, axis=0)
        var_13 = var_12.reshape([1, 7581])
        var_14 = var_3.__gt__(0)
        var_15 = paddle.tensor.creation.full_like(var_13, 80)
        var_16 = paddle.tensor.search.where(var_14, var_13, var_15)
        var_17 = var_4.reshape([-1, 4])
        var_18 = var_9.flatten()
        var_19 = paddle.tensor.manipulation.gather(var_17, var_18, axis=0)
        var_20 = var_19.reshape([1, 7581, 4])
        var_21 = paddle.nn.functional.input.one_hot(var_16, 81)
        var_22 = paddle.tensor.creation.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79])
        var_23 = paddle.tensor.search.index_select(var_21, var_22, axis=-1)
        var_24 = var_5.__mul__(var_0)
        var_25 = var_24.max(axis=-1, keepdim=True)
        var_26 = var_6.__mul__(var_0)
        var_27 = var_26.max(axis=-1, keepdim=True)
        var_28 = var_25.__add__(1e-09)
        var_29 = var_24.__truediv__(var_28)
        var_30 = var_29.__mul__(var_27)
        var_31 = var_30.max(-2)
        var_32 = var_31.unsqueeze(-1)
        var_33 = var_23.__mul__(var_32)
        return var_16, var_20, var_33


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 1, 7581], dtype=paddle.float32),
        paddle.randint(low=0, high=10, shape=[1, 1], dtype=paddle.int32),
        paddle.randint(low=0, high=10, shape=[1, 1, 1], dtype=paddle.int32),
        paddle.rand(shape=[1, 7581], dtype=paddle.float32),
        paddle.rand(shape=[1, 1, 4], dtype=paddle.float32),
        paddle.rand(shape=[1, 1, 7581], dtype=paddle.float32),
        paddle.rand(shape=[1, 1, 7581], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 1, 7581]).astype('float32'),
        np.random.randint(low=0, high=10, size=[1, 1], dtype='int32'),
        np.random.randint(low=0, high=10, size=[1, 1, 1], dtype='int32'),
        np.random.random(size=[1, 7581]).astype('float32'),
        np.random.random(size=[1, 1, 4]).astype('float32'),
        np.random.random(size=[1, 1, 7581]).astype('float32'),
        np.random.random(size=[1, 1, 7581]).astype('float32'),
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