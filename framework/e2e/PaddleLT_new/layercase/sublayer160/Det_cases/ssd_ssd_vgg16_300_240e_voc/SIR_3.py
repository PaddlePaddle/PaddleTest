# api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[16, 512, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[84],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[24],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[16, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[24, 512, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[126, 1024, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
           shape=[84, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
           shape=[24],
           dtype=paddle.float32,
        )
        self.parameter_8 = self.create_parameter(
           shape=[16],
           dtype=paddle.float32,
        )
        self.parameter_9 = self.create_parameter(
           shape=[24, 1024, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_10 = self.create_parameter(
           shape=[84, 512, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_11 = self.create_parameter(
           shape=[84],
           dtype=paddle.float32,
        )
        self.parameter_12 = self.create_parameter(
           shape=[126, 512, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_13 = self.create_parameter(
           shape=[16],
           dtype=paddle.float32,
        )
        self.parameter_14 = self.create_parameter(
           shape=[84],
           dtype=paddle.float32,
        )
        self.parameter_15 = self.create_parameter(
           shape=[16, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_16 = self.create_parameter(
           shape=[126],
           dtype=paddle.float32,
        )
        self.parameter_17 = self.create_parameter(
           shape=[24, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_18 = self.create_parameter(
           shape=[126],
           dtype=paddle.float32,
        )
        self.parameter_19 = self.create_parameter(
           shape=[84, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_20 = self.create_parameter(
           shape=[24],
           dtype=paddle.float32,
        )
        self.parameter_21 = self.create_parameter(
           shape=[126],
           dtype=paddle.float32,
        )
        self.parameter_22 = self.create_parameter(
           shape=[126, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_23 = self.create_parameter(
           shape=[16],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 512, 38, 38], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 1024, 19, 19], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [1, 512, 10, 10], dtype: paddle.float32, stop_gradient: False)
        var_3,    # (shape: [1, 256, 5, 5], dtype: paddle.float32, stop_gradient: False)
        var_4,    # (shape: [1, 256, 3, 3], dtype: paddle.float32, stop_gradient: False)
        var_5,    # (shape: [1, 256, 1, 1], dtype: paddle.float32, stop_gradient: False)
    ):
        var_6 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_0, bias=self.parameter_13, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_7 = paddle.tensor.linalg.transpose(var_6, [0, 2, 3, 1])
        var_8 = paddle.tensor.manipulation.reshape(var_7, [0, -1, 4])
        var_9 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_10, bias=self.parameter_14, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_10 = paddle.tensor.linalg.transpose(var_9, [0, 2, 3, 1])
        var_11 = paddle.tensor.manipulation.reshape(var_10, [0, -1, 21])
        var_12 = paddle.nn.functional.conv._conv_nd(var_1, self.parameter_9, bias=self.parameter_20, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_13 = paddle.tensor.linalg.transpose(var_12, [0, 2, 3, 1])
        var_14 = paddle.tensor.manipulation.reshape(var_13, [0, -1, 4])
        var_15 = paddle.nn.functional.conv._conv_nd(var_1, self.parameter_5, bias=self.parameter_21, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_16 = paddle.tensor.linalg.transpose(var_15, [0, 2, 3, 1])
        var_17 = paddle.tensor.manipulation.reshape(var_16, [0, -1, 21])
        var_18 = paddle.nn.functional.conv._conv_nd(var_2, self.parameter_4, bias=self.parameter_2, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_19 = paddle.tensor.linalg.transpose(var_18, [0, 2, 3, 1])
        var_20 = paddle.tensor.manipulation.reshape(var_19, [0, -1, 4])
        var_21 = paddle.nn.functional.conv._conv_nd(var_2, self.parameter_12, bias=self.parameter_18, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_22 = paddle.tensor.linalg.transpose(var_21, [0, 2, 3, 1])
        var_23 = paddle.tensor.manipulation.reshape(var_22, [0, -1, 21])
        var_24 = paddle.nn.functional.conv._conv_nd(var_3, self.parameter_17, bias=self.parameter_7, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_25 = paddle.tensor.linalg.transpose(var_24, [0, 2, 3, 1])
        var_26 = paddle.tensor.manipulation.reshape(var_25, [0, -1, 4])
        var_27 = paddle.nn.functional.conv._conv_nd(var_3, self.parameter_22, bias=self.parameter_16, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_28 = paddle.tensor.linalg.transpose(var_27, [0, 2, 3, 1])
        var_29 = paddle.tensor.manipulation.reshape(var_28, [0, -1, 21])
        var_30 = paddle.nn.functional.conv._conv_nd(var_4, self.parameter_3, bias=self.parameter_23, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_31 = paddle.tensor.linalg.transpose(var_30, [0, 2, 3, 1])
        var_32 = paddle.tensor.manipulation.reshape(var_31, [0, -1, 4])
        var_33 = paddle.nn.functional.conv._conv_nd(var_4, self.parameter_6, bias=self.parameter_11, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_34 = paddle.tensor.linalg.transpose(var_33, [0, 2, 3, 1])
        var_35 = paddle.tensor.manipulation.reshape(var_34, [0, -1, 21])
        var_36 = paddle.nn.functional.conv._conv_nd(var_5, self.parameter_15, bias=self.parameter_8, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_37 = paddle.tensor.linalg.transpose(var_36, [0, 2, 3, 1])
        var_38 = paddle.tensor.manipulation.reshape(var_37, [0, -1, 4])
        var_39 = paddle.nn.functional.conv._conv_nd(var_5, self.parameter_19, bias=self.parameter_1, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_40 = paddle.tensor.linalg.transpose(var_39, [0, 2, 3, 1])
        var_41 = paddle.tensor.manipulation.reshape(var_40, [0, -1, 21])
        return var_8, var_14, var_20, var_26, var_32, var_38, var_11, var_17, var_23, var_29, var_35, var_41


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 512, 38, 38], dtype=paddle.float32),
        paddle.rand(shape=[1, 1024, 19, 19], dtype=paddle.float32),
        paddle.rand(shape=[1, 512, 10, 10], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 5, 5], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 3, 3], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 1, 1], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 512, 38, 38]).astype('float32'),
        np.random.random(size=[1, 1024, 19, 19]).astype('float32'),
        np.random.random(size=[1, 512, 10, 10]).astype('float32'),
        np.random.random(size=[1, 256, 5, 5]).astype('float32'),
        np.random.random(size=[1, 256, 3, 3]).astype('float32'),
        np.random.random(size=[1, 256, 1, 1]).astype('float32'),
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