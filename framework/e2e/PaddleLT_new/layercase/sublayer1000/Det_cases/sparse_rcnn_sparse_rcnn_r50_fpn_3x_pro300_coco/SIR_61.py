# method:flatten||api:paddle.tensor.ops.sigmoid||method:flatten||api:paddle.tensor.manipulation.concat||method:__gt__||method:all
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [1, 300, 80], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 300, 4], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [2], dtype: paddle.int32, stop_gradient: True)
    ):
        var_3 = var_0.flatten(start_axis=0, stop_axis=1)
        var_4 = paddle.tensor.ops.sigmoid(var_3)
        var_5 = var_1.flatten(start_axis=0, stop_axis=1)
        var_6 = paddle.tensor.manipulation.concat([var_2])
        var_7 = var_6.__gt__(-1)
        var_8 = var_7.all()
        return var_8, var_4, var_6, var_5


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 300, 80], dtype=paddle.float32),
        paddle.rand(shape=[1, 300, 4], dtype=paddle.float32),
        paddle.randint(low=0, high=10, shape=[2], dtype=paddle.int32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 300, 80]).astype('float32'),
        np.random.random(size=[1, 300, 4]).astype('float32'),
        np.random.randint(low=0, high=10, size=[2], dtype='int32'),
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