# api:paddle.tensor.manipulation.gather||api:paddle.tensor.manipulation.gather||method:cast||api:paddle.nn.functional.loss.binary_cross_entropy_with_logits||api:paddle.tensor.manipulation.gather||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.gather||method:__sub__||api:paddle.tensor.layer_function_generator.abs||method:sum||method:__truediv__||method:__truediv__
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [185691], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [256, 1], dtype: paddle.int64, stop_gradient: True)
        var_2,    # (shape: [185691], dtype: paddle.int32, stop_gradient: True)
        var_3,    # (shape: [8, 1], dtype: paddle.int64, stop_gradient: True)
        var_4,    # (shape: [185691, 4], dtype: paddle.float32, stop_gradient: False)
        var_5,    # (shape: [185691, 4], dtype: paddle.float32, stop_gradient: True)
    ):
        var_6 = paddle.tensor.manipulation.gather(var_0, var_1)
        var_7 = paddle.tensor.manipulation.gather(var_2, var_1)
        var_8 = var_7.cast('float32')
        var_9 = paddle.nn.functional.loss.binary_cross_entropy_with_logits(logit=var_6, label=var_8, reduction='sum')
        var_10 = paddle.tensor.manipulation.gather(var_4, var_3)
        var_11 = paddle.tensor.manipulation.concat([var_5])
        var_12 = paddle.tensor.manipulation.gather(var_11, var_3)
        var_13 = var_10.__sub__(var_12)
        var_14 = paddle.tensor.abs(var_13)
        var_15 = var_14.sum()
        var_16 = var_9.__truediv__(256)
        var_17 = var_15.__truediv__(256)
        return var_16, var_17


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[185691], dtype=paddle.float32),
        paddle.randint(low=0, high=10, shape=[256, 1], dtype=paddle.int64),
        paddle.randint(low=0, high=10, shape=[185691], dtype=paddle.int32),
        paddle.randint(low=0, high=10, shape=[8, 1], dtype=paddle.int64),
        paddle.rand(shape=[185691, 4], dtype=paddle.float32),
        paddle.rand(shape=[185691, 4], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[185691]).astype('float32'),
        np.random.randint(low=0, high=10, size=[256, 1], dtype='int64'),
        np.random.randint(low=0, high=10, size=[185691], dtype='int32'),
        np.random.randint(low=0, high=10, size=[8, 1], dtype='int64'),
        np.random.random(size=[185691, 4]).astype('float32'),
        np.random.random(size=[185691, 4]).astype('float32'),
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