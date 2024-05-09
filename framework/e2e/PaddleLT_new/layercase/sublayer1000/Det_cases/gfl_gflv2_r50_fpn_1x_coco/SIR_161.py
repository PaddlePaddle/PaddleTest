# api:paddle.tensor.creation.zeros||api:paddle.nn.functional.loss.binary_cross_entropy||method:pow||method:__mul__||method:__ge__||method:__lt__||api:paddle.tensor.logic.logical_and||method:nonzero||method:squeeze
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [950, 81], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [950], dtype: paddle.int64, stop_gradient: True)
    ):
        var_2 = paddle.tensor.creation.zeros([950, 81], dtype='float32')
        var_3 = paddle.nn.functional.loss.binary_cross_entropy(var_0, var_2, reduction='none')
        var_4 = var_0.pow(2.0)
        var_5 = var_3.__mul__(var_4)
        var_6 = var_1.__ge__(0)
        var_7 = var_1.__lt__(81)
        var_8 = paddle.tensor.logic.logical_and(var_6, var_7)
        var_9 = var_8.nonzero()
        var_10 = var_9.squeeze(1)
        return var_10, var_5


def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[950, 81], dtype=paddle.float32),
        paddle.randint(low=0, high=10, shape=[950], dtype=paddle.int64),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[950, 81]).astype('float32'),
        np.random.randint(low=0, high=10, size=[950], dtype='int64'),
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