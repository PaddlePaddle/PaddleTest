# api:paddle.tensor.manipulation.concat||api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [1, 36, 28, 50], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 36, 28, 50], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = paddle.tensor.manipulation.concat(x=[var_1, var_0], axis=1)
        var_3 = paddle.tensor.attribute.shape(var_2)
        var_4 = var_3.__getitem__(2)
        var_5 = var_3.__getitem__(3)
        var_6 = paddle.tensor.manipulation.reshape(x=var_2, shape=[0, 2, 36, var_4, var_5])
        var_7 = paddle.tensor.linalg.transpose(x=var_6, perm=[0, 2, 1, 3, 4])
        var_8 = paddle.tensor.manipulation.reshape(x=var_7, shape=[0, 72, var_4, var_5])
        return var_8


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 36, 28, 50], dtype=paddle.float32),
        paddle.rand(shape=[1, 36, 28, 50], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 36, 28, 50]).astype('float32'),
        np.random.random(size=[1, 36, 28, 50]).astype('float32'),
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