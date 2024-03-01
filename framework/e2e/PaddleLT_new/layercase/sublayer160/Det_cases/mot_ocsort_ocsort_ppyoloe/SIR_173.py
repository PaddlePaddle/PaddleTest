# api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:__getitem__||method:reshape||api:paddle.nn.functional.activation.softmax||method:transpose||api:paddle.nn.functional.conv._conv_nd||method:squeeze||api:paddle.tensor.manipulation.split||method:__neg__||method:__add__||method:__add__||api:paddle.tensor.manipulation.concat
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[1, 17, 1, 1],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [6069, 2], dtype: paddle.float32, stop_gradient: True)
        var_1,    # (shape: [1, 6069, 68], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = paddle.tensor.attribute.shape(var_1)
        var_3 = var_2.__getitem__(0)
        var_4 = var_2.__getitem__(1)
        var_5 = var_2.__getitem__(2)
        var_6 = var_1.reshape([-1, var_4, 4, 17])
        var_7 = paddle.nn.functional.activation.softmax(var_6)
        var_8 = var_7.transpose([0, 3, 1, 2])
        var_9 = paddle.nn.functional.conv._conv_nd(var_8, self.parameter_0, bias=None, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_10 = var_9.squeeze(1)
        out = paddle.tensor.manipulation.split(var_10, 2, -1)
        var_11 = out[0]
        var_12 = out[1]
        var_13 = var_11.__neg__()
        var_14 = var_13.__add__(var_0)
        var_15 = var_12.__add__(var_0)
        var_16 = paddle.tensor.manipulation.concat([var_14, var_15], -1)
        return var_16


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[6069, 2], dtype=paddle.float32),
        paddle.rand(shape=[1, 6069, 68], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[6069, 2]).astype('float32'),
        np.random.random(size=[1, 6069, 68]).astype('float32'),
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