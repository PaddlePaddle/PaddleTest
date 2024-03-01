# api:paddle.nn.functional.pooling.adaptive_avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.ops.sigmoid||method:__mul__||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.ops.sigmoid||method:__mul__||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat||api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:__getitem__||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:__getitem__||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[5],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[10],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[20],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[5, 20, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[40],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[10, 40, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
           shape=[20, 5, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
           shape=[40, 10, 1, 1],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 20, 128, 256], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 40, 64, 128], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [1, 20, 128, 256], dtype: paddle.float32, stop_gradient: False)
        var_3,    # (shape: [1, 40, 64, 128], dtype: paddle.float32, stop_gradient: False)
    ):
        var_4 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_0, output_size=1, data_format='NCHW', name=None)
        var_5 = paddle.nn.functional.conv._conv_nd(var_4, self.parameter_3, bias=self.parameter_0, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_6 = paddle.nn.functional.activation.relu(var_5)
        var_7 = paddle.nn.functional.conv._conv_nd(var_6, self.parameter_6, bias=self.parameter_2, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_8 = paddle.tensor.ops.sigmoid(var_7)
        var_9 = var_0.__mul__(var_8)
        var_10 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_1, output_size=1, data_format='NCHW', name=None)
        var_11 = paddle.nn.functional.conv._conv_nd(var_10, self.parameter_5, bias=self.parameter_1, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_12 = paddle.nn.functional.activation.relu(var_11)
        var_13 = paddle.nn.functional.conv._conv_nd(var_12, self.parameter_7, bias=self.parameter_4, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_14 = paddle.tensor.ops.sigmoid(var_13)
        var_15 = var_1.__mul__(var_14)
        var_16 = paddle.tensor.manipulation.concat([var_2, var_9], axis=1)
        var_17 = paddle.tensor.manipulation.concat([var_3, var_15], axis=1)
        var_18 = paddle.tensor.attribute.shape(var_16)
        var_19 = var_18.__getitem__(0)
        var_20 = var_18.__getitem__(2)
        var_21 = var_18.__getitem__(3)
        var_22 = paddle.tensor.manipulation.reshape(x=var_16, shape=[var_19, 2, 20, var_20, var_21])
        var_23 = paddle.tensor.linalg.transpose(x=var_22, perm=[0, 2, 1, 3, 4])
        var_24 = paddle.tensor.manipulation.reshape(x=var_23, shape=[var_19, 40, var_20, var_21])
        var_25 = paddle.tensor.attribute.shape(var_17)
        var_26 = var_25.__getitem__(0)
        var_27 = var_25.__getitem__(2)
        var_28 = var_25.__getitem__(3)
        var_29 = paddle.tensor.manipulation.reshape(x=var_17, shape=[var_26, 2, 40, var_27, var_28])
        var_30 = paddle.tensor.linalg.transpose(x=var_29, perm=[0, 2, 1, 3, 4])
        var_31 = paddle.tensor.manipulation.reshape(x=var_30, shape=[var_26, 80, var_27, var_28])
        return var_24, var_31


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 20, 128, 256], dtype=paddle.float32),
        paddle.rand(shape=[1, 40, 64, 128], dtype=paddle.float32),
        paddle.rand(shape=[1, 20, 128, 256], dtype=paddle.float32),
        paddle.rand(shape=[1, 40, 64, 128], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 20, 128, 256]).astype('float32'),
        np.random.random(size=[1, 40, 64, 128]).astype('float32'),
        np.random.random(size=[1, 20, 128, 256]).astype('float32'),
        np.random.random(size=[1, 40, 64, 128]).astype('float32'),
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