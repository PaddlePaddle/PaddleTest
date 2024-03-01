# api:paddle.nn.functional.pooling.adaptive_avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.ops.sigmoid||method:__mul__||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.ops.sigmoid||method:__mul__||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.ops.sigmoid||method:__mul__||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.ops.sigmoid||method:__mul__||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat||api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:__getitem__||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:__getitem__||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:__getitem__||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:__getitem__||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape
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
           shape=[80],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[80, 20, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[160],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[10, 40, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[160, 40, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
           shape=[20],
           dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
           shape=[40, 10, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_8 = self.create_parameter(
           shape=[40],
           dtype=paddle.float32,
        )
        self.parameter_9 = self.create_parameter(
           shape=[40, 160, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_10 = self.create_parameter(
           shape=[40],
           dtype=paddle.float32,
        )
        self.parameter_11 = self.create_parameter(
           shape=[20, 80, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_12 = self.create_parameter(
           shape=[10],
           dtype=paddle.float32,
        )
        self.parameter_13 = self.create_parameter(
           shape=[20, 5, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_14 = self.create_parameter(
           shape=[5, 20, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_15 = self.create_parameter(
           shape=[20],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 20, 128, 256], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 40, 64, 128], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [1, 80, 32, 64], dtype: paddle.float32, stop_gradient: False)
        var_3,    # (shape: [1, 160, 16, 32], dtype: paddle.float32, stop_gradient: False)
        var_4,    # (shape: [1, 20, 128, 256], dtype: paddle.float32, stop_gradient: False)
        var_5,    # (shape: [1, 40, 64, 128], dtype: paddle.float32, stop_gradient: False)
        var_6,    # (shape: [1, 80, 32, 64], dtype: paddle.float32, stop_gradient: False)
        var_7,    # (shape: [1, 160, 16, 32], dtype: paddle.float32, stop_gradient: False)
    ):
        var_8 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_0, output_size=1, data_format='NCHW', name=None)
        var_9 = paddle.nn.functional.conv._conv_nd(var_8, self.parameter_14, bias=self.parameter_0, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_10 = paddle.nn.functional.activation.relu(var_9)
        var_11 = paddle.nn.functional.conv._conv_nd(var_10, self.parameter_13, bias=self.parameter_15, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_12 = paddle.tensor.ops.sigmoid(var_11)
        var_13 = var_0.__mul__(var_12)
        var_14 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_1, output_size=1, data_format='NCHW', name=None)
        var_15 = paddle.nn.functional.conv._conv_nd(var_14, self.parameter_4, bias=self.parameter_12, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_16 = paddle.nn.functional.activation.relu(var_15)
        var_17 = paddle.nn.functional.conv._conv_nd(var_16, self.parameter_7, bias=self.parameter_10, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_18 = paddle.tensor.ops.sigmoid(var_17)
        var_19 = var_1.__mul__(var_18)
        var_20 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_2, output_size=1, data_format='NCHW', name=None)
        var_21 = paddle.nn.functional.conv._conv_nd(var_20, self.parameter_11, bias=self.parameter_6, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_22 = paddle.nn.functional.activation.relu(var_21)
        var_23 = paddle.nn.functional.conv._conv_nd(var_22, self.parameter_2, bias=self.parameter_1, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_24 = paddle.tensor.ops.sigmoid(var_23)
        var_25 = var_2.__mul__(var_24)
        var_26 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_3, output_size=1, data_format='NCHW', name=None)
        var_27 = paddle.nn.functional.conv._conv_nd(var_26, self.parameter_9, bias=self.parameter_8, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_28 = paddle.nn.functional.activation.relu(var_27)
        var_29 = paddle.nn.functional.conv._conv_nd(var_28, self.parameter_5, bias=self.parameter_3, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_30 = paddle.tensor.ops.sigmoid(var_29)
        var_31 = var_3.__mul__(var_30)
        var_32 = paddle.tensor.manipulation.concat([var_4, var_13], axis=1)
        var_33 = paddle.tensor.manipulation.concat([var_5, var_19], axis=1)
        var_34 = paddle.tensor.manipulation.concat([var_6, var_25], axis=1)
        var_35 = paddle.tensor.manipulation.concat([var_7, var_31], axis=1)
        var_36 = paddle.tensor.attribute.shape(var_32)
        var_37 = var_36.__getitem__(0)
        var_38 = var_36.__getitem__(2)
        var_39 = var_36.__getitem__(3)
        var_40 = paddle.tensor.manipulation.reshape(x=var_32, shape=[var_37, 2, 20, var_38, var_39])
        var_41 = paddle.tensor.linalg.transpose(x=var_40, perm=[0, 2, 1, 3, 4])
        var_42 = paddle.tensor.manipulation.reshape(x=var_41, shape=[var_37, 40, var_38, var_39])
        var_43 = paddle.tensor.attribute.shape(var_33)
        var_44 = var_43.__getitem__(0)
        var_45 = var_43.__getitem__(2)
        var_46 = var_43.__getitem__(3)
        var_47 = paddle.tensor.manipulation.reshape(x=var_33, shape=[var_44, 2, 40, var_45, var_46])
        var_48 = paddle.tensor.linalg.transpose(x=var_47, perm=[0, 2, 1, 3, 4])
        var_49 = paddle.tensor.manipulation.reshape(x=var_48, shape=[var_44, 80, var_45, var_46])
        var_50 = paddle.tensor.attribute.shape(var_34)
        var_51 = var_50.__getitem__(0)
        var_52 = var_50.__getitem__(2)
        var_53 = var_50.__getitem__(3)
        var_54 = paddle.tensor.manipulation.reshape(x=var_34, shape=[var_51, 2, 80, var_52, var_53])
        var_55 = paddle.tensor.linalg.transpose(x=var_54, perm=[0, 2, 1, 3, 4])
        var_56 = paddle.tensor.manipulation.reshape(x=var_55, shape=[var_51, 160, var_52, var_53])
        var_57 = paddle.tensor.attribute.shape(var_35)
        var_58 = var_57.__getitem__(0)
        var_59 = var_57.__getitem__(2)
        var_60 = var_57.__getitem__(3)
        var_61 = paddle.tensor.manipulation.reshape(x=var_35, shape=[var_58, 2, 160, var_59, var_60])
        var_62 = paddle.tensor.linalg.transpose(x=var_61, perm=[0, 2, 1, 3, 4])
        var_63 = paddle.tensor.manipulation.reshape(x=var_62, shape=[var_58, 320, var_59, var_60])
        return var_42, var_49, var_56, var_63


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 20, 128, 256], dtype=paddle.float32),
        paddle.rand(shape=[1, 40, 64, 128], dtype=paddle.float32),
        paddle.rand(shape=[1, 80, 32, 64], dtype=paddle.float32),
        paddle.rand(shape=[1, 160, 16, 32], dtype=paddle.float32),
        paddle.rand(shape=[1, 20, 128, 256], dtype=paddle.float32),
        paddle.rand(shape=[1, 40, 64, 128], dtype=paddle.float32),
        paddle.rand(shape=[1, 80, 32, 64], dtype=paddle.float32),
        paddle.rand(shape=[1, 160, 16, 32], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 20, 128, 256]).astype('float32'),
        np.random.random(size=[1, 40, 64, 128]).astype('float32'),
        np.random.random(size=[1, 80, 32, 64]).astype('float32'),
        np.random.random(size=[1, 160, 16, 32]).astype('float32'),
        np.random.random(size=[1, 20, 128, 256]).astype('float32'),
        np.random.random(size=[1, 40, 64, 128]).astype('float32'),
        np.random.random(size=[1, 80, 32, 64]).astype('float32'),
        np.random.random(size=[1, 160, 16, 32]).astype('float32'),
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