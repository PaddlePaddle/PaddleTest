# api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [6, 1], dtype: paddle.int64, stop_gradient: True)
        var_1,    # (shape: [1], dtype: paddle.int32, stop_gradient: True)
        var_2,    # (shape: [6], dtype: paddle.int32, stop_gradient: True)
        var_3,    # (shape: [6, 28, 28], dtype: paddle.int32, stop_gradient: True)
        var_4,    # (shape: [6], dtype: paddle.float32, stop_gradient: True)
    ):
        var_5 = paddle.tensor.manipulation.concat([var_0])
        var_6 = paddle.tensor.manipulation.concat([var_1])
        var_7 = paddle.tensor.manipulation.concat([var_2], axis=0)
        var_8 = paddle.tensor.manipulation.concat([var_3], axis=0)
        var_9 = paddle.tensor.manipulation.concat([var_4], axis=0)
        return var_6, var_7, var_8, var_5, var_9


def create_paddle_inputs():
    inputs = (
        paddle.randint(low=0, high=10, shape=[6, 1], dtype=paddle.int64),
        paddle.randint(low=0, high=10, shape=[1], dtype=paddle.int32),
        paddle.randint(low=0, high=10, shape=[6], dtype=paddle.int32),
        paddle.randint(low=0, high=10, shape=[6, 28, 28], dtype=paddle.int32),
        paddle.rand(shape=[6], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.randint(low=0, high=10, size=[6, 1], dtype='int64'),
        np.random.randint(low=0, high=10, size=[1], dtype='int32'),
        np.random.randint(low=0, high=10, size=[6], dtype='int32'),
        np.random.randint(low=0, high=10, size=[6, 28, 28], dtype='int32'),
        np.random.random(size=[6]).astype('float32'),
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