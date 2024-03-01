# api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.pooling.max_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.pooling.max_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.pooling.max_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.pooling.max_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.pooling.max_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.norm.normalize||method:unsqueeze||method:unsqueeze||method:unsqueeze||method:__mul__
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[128, 64, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[128, 128, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[256, 128, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[64],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[512],
           dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
           shape=[512],
           dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
           shape=[128, 512, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_8 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_9 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_10 = self.create_parameter(
           shape=[256, 128, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_11 = self.create_parameter(
           shape=[512],
           dtype=paddle.float32,
        )
        self.parameter_12 = self.create_parameter(
           shape=[128, 256, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_13 = self.create_parameter(
           shape=[64, 64, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_14 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_15 = self.create_parameter(
           shape=[512],
           dtype=paddle.float32,
        )
        self.parameter_16 = self.create_parameter(
           shape=[256, 128, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_17 = self.create_parameter(
           shape=[512],
           dtype=paddle.float32,
        )
        self.parameter_18 = self.create_parameter(
           shape=[64, 3, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_19 = self.create_parameter(
           shape=[256, 128, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_20 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_21 = self.create_parameter(
           shape=[128],
           dtype=paddle.float32,
        )
        self.parameter_22 = self.create_parameter(
           shape=[512, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_23 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_24 = self.create_parameter(
           shape=[1024, 1024, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_25 = self.create_parameter(
           shape=[128],
           dtype=paddle.float32,
        )
        self.parameter_26 = self.create_parameter(
           shape=[128],
           dtype=paddle.float32,
        )
        self.parameter_27 = self.create_parameter(
           shape=[512],
           dtype=paddle.float32,
        )
        self.parameter_28 = self.create_parameter(
           shape=[512, 512, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_29 = self.create_parameter(
           shape=[128],
           dtype=paddle.float32,
        )
        self.parameter_30 = self.create_parameter(
           shape=[1024, 512, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_31 = self.create_parameter(
           shape=[512, 512, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_32 = self.create_parameter(
           shape=[128],
           dtype=paddle.float32,
        )
        self.parameter_33 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_34 = self.create_parameter(
           shape=[64],
           dtype=paddle.float32,
        )
        self.parameter_35 = self.create_parameter(
           shape=[1024],
           dtype=paddle.float32,
        )
        self.parameter_36 = self.create_parameter(
           shape=[512, 512, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_37 = self.create_parameter(
           shape=[256, 1024, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_38 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_39 = self.create_parameter(
           shape=[512],
           dtype=paddle.float32,
        )
        self.parameter_40 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_41 = self.create_parameter(
           shape=[512],
           dtype=paddle.float32,
        )
        self.parameter_42 = self.create_parameter(
           shape=[512, 512, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_43 = self.create_parameter(
           shape=[512, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_44 = self.create_parameter(
           shape=[1024],
           dtype=paddle.float32,
        )
        self.parameter_45 = self.create_parameter(
           shape=[128, 256, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_46 = self.create_parameter(
           shape=[512, 512, 3, 3],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 3, 300, 300], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_18, bias=self.parameter_4, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_2 = paddle.nn.functional.activation.relu(var_1)
        var_3 = paddle.nn.functional.conv._conv_nd(var_2, self.parameter_13, bias=self.parameter_34, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_4 = paddle.nn.functional.activation.relu(var_3)
        var_5 = paddle.nn.functional.pooling.max_pool2d(var_4, kernel_size=2, stride=2, padding=0, return_mask=False, ceil_mode=True, data_format='NCHW', name=None)
        var_6 = paddle.nn.functional.conv._conv_nd(var_5, self.parameter_0, bias=self.parameter_29, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_7 = paddle.nn.functional.activation.relu(var_6)
        var_8 = paddle.nn.functional.conv._conv_nd(var_7, self.parameter_1, bias=self.parameter_25, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_9 = paddle.nn.functional.activation.relu(var_8)
        var_10 = paddle.nn.functional.pooling.max_pool2d(var_9, kernel_size=2, stride=2, padding=0, return_mask=False, ceil_mode=True, data_format='NCHW', name=None)
        var_11 = paddle.nn.functional.conv._conv_nd(var_10, self.parameter_10, bias=self.parameter_20, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_12 = paddle.nn.functional.activation.relu(var_11)
        var_13 = paddle.nn.functional.conv._conv_nd(var_12, self.parameter_33, bias=self.parameter_40, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_14 = paddle.nn.functional.activation.relu(var_13)
        var_15 = paddle.nn.functional.conv._conv_nd(var_14, self.parameter_8, bias=self.parameter_38, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_16 = paddle.nn.functional.activation.relu(var_15)
        var_17 = paddle.nn.functional.pooling.max_pool2d(var_16, kernel_size=2, stride=2, padding=0, return_mask=False, ceil_mode=True, data_format='NCHW', name=None)
        var_18 = paddle.nn.functional.conv._conv_nd(var_17, self.parameter_22, bias=self.parameter_15, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_19 = paddle.nn.functional.activation.relu(var_18)
        var_20 = paddle.nn.functional.conv._conv_nd(var_19, self.parameter_31, bias=self.parameter_5, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_21 = paddle.nn.functional.activation.relu(var_20)
        var_22 = paddle.nn.functional.conv._conv_nd(var_21, self.parameter_46, bias=self.parameter_27, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_23 = paddle.nn.functional.activation.relu(var_22)
        var_24 = paddle.nn.functional.pooling.max_pool2d(var_23, kernel_size=2, stride=2, padding=0, return_mask=False, ceil_mode=True, data_format='NCHW', name=None)
        var_25 = paddle.nn.functional.conv._conv_nd(var_24, self.parameter_28, bias=self.parameter_11, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_26 = paddle.nn.functional.activation.relu(var_25)
        var_27 = paddle.nn.functional.conv._conv_nd(var_26, self.parameter_36, bias=self.parameter_41, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_28 = paddle.nn.functional.activation.relu(var_27)
        var_29 = paddle.nn.functional.conv._conv_nd(var_28, self.parameter_42, bias=self.parameter_39, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_30 = paddle.nn.functional.activation.relu(var_29)
        var_31 = paddle.nn.functional.pooling.max_pool2d(var_30, kernel_size=3, stride=1, padding=1, return_mask=False, ceil_mode=True, data_format='NCHW', name=None)
        var_32 = paddle.nn.functional.conv._conv_nd(var_31, self.parameter_30, bias=self.parameter_35, stride=[1, 1], padding=[6, 6], padding_algorithm='EXPLICIT', dilation=[6, 6], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_33 = paddle.nn.functional.activation.relu(var_32)
        var_34 = paddle.nn.functional.conv._conv_nd(var_33, self.parameter_24, bias=self.parameter_44, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_35 = paddle.nn.functional.activation.relu(var_34)
        var_36 = paddle.nn.functional.conv._conv_nd(var_35, self.parameter_37, bias=self.parameter_14, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_37 = paddle.nn.functional.activation.relu(var_36)
        var_38 = paddle.nn.functional.conv._conv_nd(var_37, self.parameter_43, bias=self.parameter_6, stride=[2, 2], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_39 = paddle.nn.functional.activation.relu(var_38)
        var_40 = paddle.nn.functional.conv._conv_nd(var_39, self.parameter_7, bias=self.parameter_26, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_41 = paddle.nn.functional.activation.relu(var_40)
        var_42 = paddle.nn.functional.conv._conv_nd(var_41, self.parameter_16, bias=self.parameter_23, stride=[2, 2], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_43 = paddle.nn.functional.activation.relu(var_42)
        var_44 = paddle.nn.functional.conv._conv_nd(var_43, self.parameter_45, bias=self.parameter_32, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_45 = paddle.nn.functional.activation.relu(var_44)
        var_46 = paddle.nn.functional.conv._conv_nd(var_45, self.parameter_19, bias=self.parameter_9, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_47 = paddle.nn.functional.activation.relu(var_46)
        var_48 = paddle.nn.functional.conv._conv_nd(var_47, self.parameter_12, bias=self.parameter_21, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_49 = paddle.nn.functional.activation.relu(var_48)
        var_50 = paddle.nn.functional.conv._conv_nd(var_49, self.parameter_2, bias=self.parameter_3, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_51 = paddle.nn.functional.activation.relu(var_50)
        var_52 = paddle.nn.functional.norm.normalize(var_23, axis=1, epsilon=1e-10)
        var_53 = self.parameter_17.unsqueeze(0)
        var_54 = var_53.unsqueeze(2)
        var_55 = var_54.unsqueeze(3)
        var_56 = var_55.__mul__(var_52)
        return var_56, var_35, var_39, var_43, var_47, var_51


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 3, 300, 300], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 3, 300, 300]).astype('float32'),
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