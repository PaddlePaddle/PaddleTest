# api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.hardsigmoid||method:__mul__||method:__add__||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.hardsigmoid||method:__mul__||method:__add__||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.hardsigmoid||method:__mul__||method:__add__||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.hardsigmoid||method:__mul__||method:__add__||api:paddle.nn.functional.common.upsample||method:__add__||api:paddle.nn.functional.common.upsample||method:__add__||api:paddle.nn.functional.common.upsample||method:__add__||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.hardsigmoid||method:__mul__||method:__add__||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.hardsigmoid||method:__mul__||method:__add__||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.hardsigmoid||method:__mul__||method:__add__||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.hardsigmoid||method:__mul__||method:__add__||api:paddle.nn.functional.common.upsample||api:paddle.nn.functional.common.upsample||api:paddle.nn.functional.common.upsample||api:paddle.tensor.manipulation.concat
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[96],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[96, 56, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[6],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[96, 480, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[96],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[96, 24, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
           shape=[24, 6, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
           shape=[24, 96, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_8 = self.create_parameter(
           shape=[24, 96, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_9 = self.create_parameter(
           shape=[96, 24, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_10 = self.create_parameter(
           shape=[24],
           dtype=paddle.float32,
        )
        self.parameter_11 = self.create_parameter(
           shape=[6],
           dtype=paddle.float32,
        )
        self.parameter_12 = self.create_parameter(
           shape=[24, 96, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_13 = self.create_parameter(
           shape=[96, 24, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_14 = self.create_parameter(
           shape=[24, 96, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_15 = self.create_parameter(
           shape=[96],
           dtype=paddle.float32,
        )
        self.parameter_16 = self.create_parameter(
           shape=[24, 6, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_17 = self.create_parameter(
           shape=[24, 96, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_18 = self.create_parameter(
           shape=[24],
           dtype=paddle.float32,
        )
        self.parameter_19 = self.create_parameter(
           shape=[24],
           dtype=paddle.float32,
        )
        self.parameter_20 = self.create_parameter(
           shape=[24, 6, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_21 = self.create_parameter(
           shape=[96],
           dtype=paddle.float32,
        )
        self.parameter_22 = self.create_parameter(
           shape=[24, 6, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_23 = self.create_parameter(
           shape=[24],
           dtype=paddle.float32,
        )
        self.parameter_24 = self.create_parameter(
           shape=[24],
           dtype=paddle.float32,
        )
        self.parameter_25 = self.create_parameter(
           shape=[96, 16, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_26 = self.create_parameter(
           shape=[96, 24, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_27 = self.create_parameter(
           shape=[6, 24, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_28 = self.create_parameter(
           shape=[24, 96, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_29 = self.create_parameter(
           shape=[6, 24, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_30 = self.create_parameter(
           shape=[6],
           dtype=paddle.float32,
        )
        self.parameter_31 = self.create_parameter(
           shape=[6],
           dtype=paddle.float32,
        )
        self.parameter_32 = self.create_parameter(
           shape=[24, 96, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_33 = self.create_parameter(
           shape=[24],
           dtype=paddle.float32,
        )
        self.parameter_34 = self.create_parameter(
           shape=[96, 24, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_35 = self.create_parameter(
           shape=[24],
           dtype=paddle.float32,
        )
        self.parameter_36 = self.create_parameter(
           shape=[6, 24, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_37 = self.create_parameter(
           shape=[6, 24, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_38 = self.create_parameter(
           shape=[24],
           dtype=paddle.float32,
        )
        self.parameter_39 = self.create_parameter(
           shape=[24, 96, 1, 1],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [2, 16, 240, 240], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [2, 24, 120, 120], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [2, 56, 60, 60], dtype: paddle.float32, stop_gradient: False)
        var_3,    # (shape: [2, 480, 30, 30], dtype: paddle.float32, stop_gradient: False)
    ):
        var_4 = paddle.nn.functional.conv._conv_nd(var_3, self.parameter_3, bias=None, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_5 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_4, output_size=1, data_format='NCHW', name=None)
        var_6 = paddle.nn.functional.conv._conv_nd(var_5, self.parameter_12, bias=self.parameter_33, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_7 = paddle.nn.functional.activation.relu(var_6)
        var_8 = paddle.nn.functional.conv._conv_nd(var_7, self.parameter_26, bias=self.parameter_0, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_9 = paddle.nn.functional.activation.hardsigmoid(var_8, slope=0.2, offset=0.5)
        var_10 = var_4.__mul__(var_9)
        var_11 = var_4.__add__(var_10)
        var_12 = paddle.nn.functional.conv._conv_nd(var_2, self.parameter_1, bias=None, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_13 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_12, output_size=1, data_format='NCHW', name=None)
        var_14 = paddle.nn.functional.conv._conv_nd(var_13, self.parameter_14, bias=self.parameter_24, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_15 = paddle.nn.functional.activation.relu(var_14)
        var_16 = paddle.nn.functional.conv._conv_nd(var_15, self.parameter_34, bias=self.parameter_21, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_17 = paddle.nn.functional.activation.hardsigmoid(var_16, slope=0.2, offset=0.5)
        var_18 = var_12.__mul__(var_17)
        var_19 = var_12.__add__(var_18)
        var_20 = paddle.nn.functional.conv._conv_nd(var_1, self.parameter_5, bias=None, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_21 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_20, output_size=1, data_format='NCHW', name=None)
        var_22 = paddle.nn.functional.conv._conv_nd(var_21, self.parameter_39, bias=self.parameter_35, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_23 = paddle.nn.functional.activation.relu(var_22)
        var_24 = paddle.nn.functional.conv._conv_nd(var_23, self.parameter_13, bias=self.parameter_15, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_25 = paddle.nn.functional.activation.hardsigmoid(var_24, slope=0.2, offset=0.5)
        var_26 = var_20.__mul__(var_25)
        var_27 = var_20.__add__(var_26)
        var_28 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_25, bias=None, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_29 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_28, output_size=1, data_format='NCHW', name=None)
        var_30 = paddle.nn.functional.conv._conv_nd(var_29, self.parameter_8, bias=self.parameter_19, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_31 = paddle.nn.functional.activation.relu(var_30)
        var_32 = paddle.nn.functional.conv._conv_nd(var_31, self.parameter_9, bias=self.parameter_4, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_33 = paddle.nn.functional.activation.hardsigmoid(var_32, slope=0.2, offset=0.5)
        var_34 = var_28.__mul__(var_33)
        var_35 = var_28.__add__(var_34)
        var_36 = paddle.nn.functional.common.upsample(var_11, scale_factor=2, mode='nearest', align_mode=1)
        var_37 = var_19.__add__(var_36)
        var_38 = paddle.nn.functional.common.upsample(var_37, scale_factor=2, mode='nearest', align_mode=1)
        var_39 = var_27.__add__(var_38)
        var_40 = paddle.nn.functional.common.upsample(var_39, scale_factor=2, mode='nearest', align_mode=1)
        var_41 = var_35.__add__(var_40)
        var_42 = paddle.nn.functional.conv._conv_nd(var_11, self.parameter_28, bias=None, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_43 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_42, output_size=1, data_format='NCHW', name=None)
        var_44 = paddle.nn.functional.conv._conv_nd(var_43, self.parameter_27, bias=self.parameter_2, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_45 = paddle.nn.functional.activation.relu(var_44)
        var_46 = paddle.nn.functional.conv._conv_nd(var_45, self.parameter_6, bias=self.parameter_18, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_47 = paddle.nn.functional.activation.hardsigmoid(var_46, slope=0.2, offset=0.5)
        var_48 = var_42.__mul__(var_47)
        var_49 = var_42.__add__(var_48)
        var_50 = paddle.nn.functional.conv._conv_nd(var_37, self.parameter_32, bias=None, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_51 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_50, output_size=1, data_format='NCHW', name=None)
        var_52 = paddle.nn.functional.conv._conv_nd(var_51, self.parameter_37, bias=self.parameter_31, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_53 = paddle.nn.functional.activation.relu(var_52)
        var_54 = paddle.nn.functional.conv._conv_nd(var_53, self.parameter_20, bias=self.parameter_38, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_55 = paddle.nn.functional.activation.hardsigmoid(var_54, slope=0.2, offset=0.5)
        var_56 = var_50.__mul__(var_55)
        var_57 = var_50.__add__(var_56)
        var_58 = paddle.nn.functional.conv._conv_nd(var_39, self.parameter_7, bias=None, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_59 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_58, output_size=1, data_format='NCHW', name=None)
        var_60 = paddle.nn.functional.conv._conv_nd(var_59, self.parameter_29, bias=self.parameter_30, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_61 = paddle.nn.functional.activation.relu(var_60)
        var_62 = paddle.nn.functional.conv._conv_nd(var_61, self.parameter_22, bias=self.parameter_23, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_63 = paddle.nn.functional.activation.hardsigmoid(var_62, slope=0.2, offset=0.5)
        var_64 = var_58.__mul__(var_63)
        var_65 = var_58.__add__(var_64)
        var_66 = paddle.nn.functional.conv._conv_nd(var_41, self.parameter_17, bias=None, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_67 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_66, output_size=1, data_format='NCHW', name=None)
        var_68 = paddle.nn.functional.conv._conv_nd(var_67, self.parameter_36, bias=self.parameter_11, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_69 = paddle.nn.functional.activation.relu(var_68)
        var_70 = paddle.nn.functional.conv._conv_nd(var_69, self.parameter_16, bias=self.parameter_10, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_71 = paddle.nn.functional.activation.hardsigmoid(var_70, slope=0.2, offset=0.5)
        var_72 = var_66.__mul__(var_71)
        var_73 = var_66.__add__(var_72)
        var_74 = paddle.nn.functional.common.upsample(var_49, scale_factor=8, mode='nearest', align_mode=1)
        var_75 = paddle.nn.functional.common.upsample(var_57, scale_factor=4, mode='nearest', align_mode=1)
        var_76 = paddle.nn.functional.common.upsample(var_65, scale_factor=2, mode='nearest', align_mode=1)
        var_77 = paddle.tensor.manipulation.concat([var_74, var_75, var_76, var_73], axis=1)
        return var_77


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[2, 16, 240, 240], dtype=paddle.float32),
        paddle.rand(shape=[2, 24, 120, 120], dtype=paddle.float32),
        paddle.rand(shape=[2, 56, 60, 60], dtype=paddle.float32),
        paddle.rand(shape=[2, 480, 30, 30], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[2, 16, 240, 240]).astype('float32'),
        np.random.random(size=[2, 24, 120, 120]).astype('float32'),
        np.random.random(size=[2, 56, 60, 60]).astype('float32'),
        np.random.random(size=[2, 480, 30, 30]).astype('float32'),
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