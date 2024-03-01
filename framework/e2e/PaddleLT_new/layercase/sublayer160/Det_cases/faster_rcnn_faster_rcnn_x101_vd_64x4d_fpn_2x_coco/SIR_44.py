# api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.concat||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.concat||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.concat
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [1, 3, 176, 264], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 3, 88, 132], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [1, 3, 44, 66], dtype: paddle.float32, stop_gradient: False)
        var_3,    # (shape: [1, 3, 22, 33], dtype: paddle.float32, stop_gradient: False)
        var_4,    # (shape: [1, 3, 11, 17], dtype: paddle.float32, stop_gradient: False)
        var_5,    # (shape: [1, 12, 176, 264], dtype: paddle.float32, stop_gradient: False)
        var_6,    # (shape: [1, 12, 88, 132], dtype: paddle.float32, stop_gradient: False)
        var_7,    # (shape: [1, 12, 44, 66], dtype: paddle.float32, stop_gradient: False)
        var_8,    # (shape: [1, 12, 22, 33], dtype: paddle.float32, stop_gradient: False)
        var_9,    # (shape: [1, 12, 11, 17], dtype: paddle.float32, stop_gradient: False)
        var_10,    # (shape: [139392, 4], dtype: paddle.float32, stop_gradient: True)
        var_11,    # (shape: [34848, 4], dtype: paddle.float32, stop_gradient: True)
        var_12,    # (shape: [8712, 4], dtype: paddle.float32, stop_gradient: True)
        var_13,    # (shape: [2178, 4], dtype: paddle.float32, stop_gradient: True)
        var_14,    # (shape: [561, 4], dtype: paddle.float32, stop_gradient: True)
    ):
        var_15 = paddle.tensor.manipulation.reshape(var_10, shape=(-1, 4,))
        var_16 = paddle.tensor.manipulation.reshape(var_11, shape=(-1, 4,))
        var_17 = paddle.tensor.manipulation.reshape(var_12, shape=(-1, 4,))
        var_18 = paddle.tensor.manipulation.reshape(var_13, shape=(-1, 4,))
        var_19 = paddle.tensor.manipulation.reshape(var_14, shape=(-1, 4,))
        var_20 = paddle.tensor.manipulation.concat([var_15, var_16, var_17, var_18, var_19])
        var_21 = paddle.tensor.linalg.transpose(var_0, perm=[0, 2, 3, 1])
        var_22 = paddle.tensor.manipulation.reshape(var_21, shape=(1, -1, 1,))
        var_23 = paddle.tensor.linalg.transpose(var_1, perm=[0, 2, 3, 1])
        var_24 = paddle.tensor.manipulation.reshape(var_23, shape=(1, -1, 1,))
        var_25 = paddle.tensor.linalg.transpose(var_2, perm=[0, 2, 3, 1])
        var_26 = paddle.tensor.manipulation.reshape(var_25, shape=(1, -1, 1,))
        var_27 = paddle.tensor.linalg.transpose(var_3, perm=[0, 2, 3, 1])
        var_28 = paddle.tensor.manipulation.reshape(var_27, shape=(1, -1, 1,))
        var_29 = paddle.tensor.linalg.transpose(var_4, perm=[0, 2, 3, 1])
        var_30 = paddle.tensor.manipulation.reshape(var_29, shape=(1, -1, 1,))
        var_31 = paddle.tensor.manipulation.concat([var_22, var_24, var_26, var_28, var_30], axis=1)
        var_32 = paddle.tensor.linalg.transpose(var_5, perm=[0, 2, 3, 1])
        var_33 = paddle.tensor.manipulation.reshape(var_32, shape=(1, -1, 4,))
        var_34 = paddle.tensor.linalg.transpose(var_6, perm=[0, 2, 3, 1])
        var_35 = paddle.tensor.manipulation.reshape(var_34, shape=(1, -1, 4,))
        var_36 = paddle.tensor.linalg.transpose(var_7, perm=[0, 2, 3, 1])
        var_37 = paddle.tensor.manipulation.reshape(var_36, shape=(1, -1, 4,))
        var_38 = paddle.tensor.linalg.transpose(var_8, perm=[0, 2, 3, 1])
        var_39 = paddle.tensor.manipulation.reshape(var_38, shape=(1, -1, 4,))
        var_40 = paddle.tensor.linalg.transpose(var_9, perm=[0, 2, 3, 1])
        var_41 = paddle.tensor.manipulation.reshape(var_40, shape=(1, -1, 4,))
        var_42 = paddle.tensor.manipulation.concat([var_33, var_35, var_37, var_39, var_41], axis=1)
        return var_20, var_31, var_42


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 3, 176, 264], dtype=paddle.float32),
        paddle.rand(shape=[1, 3, 88, 132], dtype=paddle.float32),
        paddle.rand(shape=[1, 3, 44, 66], dtype=paddle.float32),
        paddle.rand(shape=[1, 3, 22, 33], dtype=paddle.float32),
        paddle.rand(shape=[1, 3, 11, 17], dtype=paddle.float32),
        paddle.rand(shape=[1, 12, 176, 264], dtype=paddle.float32),
        paddle.rand(shape=[1, 12, 88, 132], dtype=paddle.float32),
        paddle.rand(shape=[1, 12, 44, 66], dtype=paddle.float32),
        paddle.rand(shape=[1, 12, 22, 33], dtype=paddle.float32),
        paddle.rand(shape=[1, 12, 11, 17], dtype=paddle.float32),
        paddle.rand(shape=[139392, 4], dtype=paddle.float32),
        paddle.rand(shape=[34848, 4], dtype=paddle.float32),
        paddle.rand(shape=[8712, 4], dtype=paddle.float32),
        paddle.rand(shape=[2178, 4], dtype=paddle.float32),
        paddle.rand(shape=[561, 4], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 3, 176, 264]).astype('float32'),
        np.random.random(size=[1, 3, 88, 132]).astype('float32'),
        np.random.random(size=[1, 3, 44, 66]).astype('float32'),
        np.random.random(size=[1, 3, 22, 33]).astype('float32'),
        np.random.random(size=[1, 3, 11, 17]).astype('float32'),
        np.random.random(size=[1, 12, 176, 264]).astype('float32'),
        np.random.random(size=[1, 12, 88, 132]).astype('float32'),
        np.random.random(size=[1, 12, 44, 66]).astype('float32'),
        np.random.random(size=[1, 12, 22, 33]).astype('float32'),
        np.random.random(size=[1, 12, 11, 17]).astype('float32'),
        np.random.random(size=[139392, 4]).astype('float32'),
        np.random.random(size=[34848, 4]).astype('float32'),
        np.random.random(size=[8712, 4]).astype('float32'),
        np.random.random(size=[2178, 4]).astype('float32'),
        np.random.random(size=[561, 4]).astype('float32'),
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