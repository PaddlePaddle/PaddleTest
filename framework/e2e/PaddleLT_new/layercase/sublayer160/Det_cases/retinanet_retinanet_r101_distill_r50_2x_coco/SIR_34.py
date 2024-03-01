# api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[36],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[720, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_8 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_9 = self.create_parameter(
           shape=[36, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_10 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_11 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_12 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_13 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_14 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_15 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_16 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_17 = self.create_parameter(
           shape=[720],
           dtype=paddle.float32,
        )
        self.parameter_18 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_19 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 256, 88, 132], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 256, 44, 66], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [1, 256, 22, 33], dtype: paddle.float32, stop_gradient: False)
        var_3,    # (shape: [1, 256, 11, 17], dtype: paddle.float32, stop_gradient: False)
        var_4,    # (shape: [1, 256, 6, 9], dtype: paddle.float32, stop_gradient: False)
    ):
        var_5 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_6, bias=self.parameter_0, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_6 = paddle.nn.functional.activation.relu(var_5)
        var_7 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_4, bias=self.parameter_18, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_8 = paddle.nn.functional.activation.relu(var_7)
        var_9 = paddle.nn.functional.conv._conv_nd(var_6, self.parameter_19, bias=self.parameter_5, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_10 = paddle.nn.functional.activation.relu(var_9)
        var_11 = paddle.nn.functional.conv._conv_nd(var_8, self.parameter_3, bias=self.parameter_14, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_12 = paddle.nn.functional.activation.relu(var_11)
        var_13 = paddle.nn.functional.conv._conv_nd(var_10, self.parameter_11, bias=self.parameter_15, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_14 = paddle.nn.functional.activation.relu(var_13)
        var_15 = paddle.nn.functional.conv._conv_nd(var_12, self.parameter_16, bias=self.parameter_12, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_16 = paddle.nn.functional.activation.relu(var_15)
        var_17 = paddle.nn.functional.conv._conv_nd(var_14, self.parameter_8, bias=self.parameter_13, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_18 = paddle.nn.functional.activation.relu(var_17)
        var_19 = paddle.nn.functional.conv._conv_nd(var_16, self.parameter_10, bias=self.parameter_7, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_20 = paddle.nn.functional.activation.relu(var_19)
        var_21 = paddle.nn.functional.conv._conv_nd(var_18, self.parameter_2, bias=self.parameter_17, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_22 = paddle.nn.functional.conv._conv_nd(var_20, self.parameter_9, bias=self.parameter_1, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_23 = paddle.nn.functional.conv._conv_nd(var_1, self.parameter_6, bias=self.parameter_0, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_24 = paddle.nn.functional.activation.relu(var_23)
        var_25 = paddle.nn.functional.conv._conv_nd(var_1, self.parameter_4, bias=self.parameter_18, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_26 = paddle.nn.functional.activation.relu(var_25)
        var_27 = paddle.nn.functional.conv._conv_nd(var_24, self.parameter_19, bias=self.parameter_5, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_28 = paddle.nn.functional.activation.relu(var_27)
        var_29 = paddle.nn.functional.conv._conv_nd(var_26, self.parameter_3, bias=self.parameter_14, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_30 = paddle.nn.functional.activation.relu(var_29)
        var_31 = paddle.nn.functional.conv._conv_nd(var_28, self.parameter_11, bias=self.parameter_15, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_32 = paddle.nn.functional.activation.relu(var_31)
        var_33 = paddle.nn.functional.conv._conv_nd(var_30, self.parameter_16, bias=self.parameter_12, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_34 = paddle.nn.functional.activation.relu(var_33)
        var_35 = paddle.nn.functional.conv._conv_nd(var_32, self.parameter_8, bias=self.parameter_13, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_36 = paddle.nn.functional.activation.relu(var_35)
        var_37 = paddle.nn.functional.conv._conv_nd(var_34, self.parameter_10, bias=self.parameter_7, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_38 = paddle.nn.functional.activation.relu(var_37)
        var_39 = paddle.nn.functional.conv._conv_nd(var_36, self.parameter_2, bias=self.parameter_17, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_40 = paddle.nn.functional.conv._conv_nd(var_38, self.parameter_9, bias=self.parameter_1, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_41 = paddle.nn.functional.conv._conv_nd(var_2, self.parameter_6, bias=self.parameter_0, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_42 = paddle.nn.functional.activation.relu(var_41)
        var_43 = paddle.nn.functional.conv._conv_nd(var_2, self.parameter_4, bias=self.parameter_18, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_44 = paddle.nn.functional.activation.relu(var_43)
        var_45 = paddle.nn.functional.conv._conv_nd(var_42, self.parameter_19, bias=self.parameter_5, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_46 = paddle.nn.functional.activation.relu(var_45)
        var_47 = paddle.nn.functional.conv._conv_nd(var_44, self.parameter_3, bias=self.parameter_14, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_48 = paddle.nn.functional.activation.relu(var_47)
        var_49 = paddle.nn.functional.conv._conv_nd(var_46, self.parameter_11, bias=self.parameter_15, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_50 = paddle.nn.functional.activation.relu(var_49)
        var_51 = paddle.nn.functional.conv._conv_nd(var_48, self.parameter_16, bias=self.parameter_12, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_52 = paddle.nn.functional.activation.relu(var_51)
        var_53 = paddle.nn.functional.conv._conv_nd(var_50, self.parameter_8, bias=self.parameter_13, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_54 = paddle.nn.functional.activation.relu(var_53)
        var_55 = paddle.nn.functional.conv._conv_nd(var_52, self.parameter_10, bias=self.parameter_7, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_56 = paddle.nn.functional.activation.relu(var_55)
        var_57 = paddle.nn.functional.conv._conv_nd(var_54, self.parameter_2, bias=self.parameter_17, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_58 = paddle.nn.functional.conv._conv_nd(var_56, self.parameter_9, bias=self.parameter_1, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_59 = paddle.nn.functional.conv._conv_nd(var_3, self.parameter_6, bias=self.parameter_0, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_60 = paddle.nn.functional.activation.relu(var_59)
        var_61 = paddle.nn.functional.conv._conv_nd(var_3, self.parameter_4, bias=self.parameter_18, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_62 = paddle.nn.functional.activation.relu(var_61)
        var_63 = paddle.nn.functional.conv._conv_nd(var_60, self.parameter_19, bias=self.parameter_5, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_64 = paddle.nn.functional.activation.relu(var_63)
        var_65 = paddle.nn.functional.conv._conv_nd(var_62, self.parameter_3, bias=self.parameter_14, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_66 = paddle.nn.functional.activation.relu(var_65)
        var_67 = paddle.nn.functional.conv._conv_nd(var_64, self.parameter_11, bias=self.parameter_15, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_68 = paddle.nn.functional.activation.relu(var_67)
        var_69 = paddle.nn.functional.conv._conv_nd(var_66, self.parameter_16, bias=self.parameter_12, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_70 = paddle.nn.functional.activation.relu(var_69)
        var_71 = paddle.nn.functional.conv._conv_nd(var_68, self.parameter_8, bias=self.parameter_13, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_72 = paddle.nn.functional.activation.relu(var_71)
        var_73 = paddle.nn.functional.conv._conv_nd(var_70, self.parameter_10, bias=self.parameter_7, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_74 = paddle.nn.functional.activation.relu(var_73)
        var_75 = paddle.nn.functional.conv._conv_nd(var_72, self.parameter_2, bias=self.parameter_17, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_76 = paddle.nn.functional.conv._conv_nd(var_74, self.parameter_9, bias=self.parameter_1, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_77 = paddle.nn.functional.conv._conv_nd(var_4, self.parameter_6, bias=self.parameter_0, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_78 = paddle.nn.functional.activation.relu(var_77)
        var_79 = paddle.nn.functional.conv._conv_nd(var_4, self.parameter_4, bias=self.parameter_18, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_80 = paddle.nn.functional.activation.relu(var_79)
        var_81 = paddle.nn.functional.conv._conv_nd(var_78, self.parameter_19, bias=self.parameter_5, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_82 = paddle.nn.functional.activation.relu(var_81)
        var_83 = paddle.nn.functional.conv._conv_nd(var_80, self.parameter_3, bias=self.parameter_14, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_84 = paddle.nn.functional.activation.relu(var_83)
        var_85 = paddle.nn.functional.conv._conv_nd(var_82, self.parameter_11, bias=self.parameter_15, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_86 = paddle.nn.functional.activation.relu(var_85)
        var_87 = paddle.nn.functional.conv._conv_nd(var_84, self.parameter_16, bias=self.parameter_12, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_88 = paddle.nn.functional.activation.relu(var_87)
        var_89 = paddle.nn.functional.conv._conv_nd(var_86, self.parameter_8, bias=self.parameter_13, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_90 = paddle.nn.functional.activation.relu(var_89)
        var_91 = paddle.nn.functional.conv._conv_nd(var_88, self.parameter_10, bias=self.parameter_7, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_92 = paddle.nn.functional.activation.relu(var_91)
        var_93 = paddle.nn.functional.conv._conv_nd(var_90, self.parameter_2, bias=self.parameter_17, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_94 = paddle.nn.functional.conv._conv_nd(var_92, self.parameter_9, bias=self.parameter_1, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        return var_21, var_39, var_57, var_75, var_93, var_22, var_40, var_58, var_76, var_94


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 256, 88, 132], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 44, 66], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 22, 33], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 11, 17], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 6, 9], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 256, 88, 132]).astype('float32'),
        np.random.random(size=[1, 256, 44, 66]).astype('float32'),
        np.random.random(size=[1, 256, 22, 33]).astype('float32'),
        np.random.random(size=[1, 256, 11, 17]).astype('float32'),
        np.random.random(size=[1, 256, 6, 9]).astype('float32'),
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