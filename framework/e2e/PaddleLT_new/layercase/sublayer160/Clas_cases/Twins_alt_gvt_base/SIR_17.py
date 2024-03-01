# method:reshape||method:transpose||api:paddle.nn.functional.common.linear||method:reshape||method:transpose||method:__getitem__||method:__getitem__||method:__getitem__||method:transpose
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[192, 576],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[576],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [43, 784, 192], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = var_0.reshape([43, 4, 7, 4, 7, 192])
        var_2 = var_1.transpose([0, 1, 3, 2, 4, 5])
        var_3 = paddle.nn.functional.common.linear(x=var_2, weight=self.parameter_0, bias=self.parameter_1, name=None)
        var_4 = var_3.reshape([43, 16, 49, 3, 6, 32])
        var_5 = var_4.transpose([3, 0, 1, 4, 2, 5])
        var_6 = var_5.__getitem__(0)
        var_7 = var_5.__getitem__(1)
        var_8 = var_5.__getitem__(2)
        var_9 = var_7.transpose([0, 1, 2, 4, 3])
        return var_6, var_9, var_8


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[43, 784, 192], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[43, 784, 192]).astype('float32'),
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