# 性能debug
import paddle
import unittest
import numpy as np
import timeit

class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[384],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[384],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[384],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[384, 768],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[768],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[384],
           dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
           shape=[384, 384, 2, 2],
           dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
           shape=[384, 384],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [43, 196, 384], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1 = paddle.nn.functional.common.linear(x=var_0, weight=self.parameter_7, bias=self.parameter_5, name=None)
        var_2 = var_1.reshape([43, 196, 12, 32])
        var_3 = var_2.transpose([0, 2, 1, 3])
        var_4 = var_0.transpose([0, 2, 1])
        var_5 = var_4.reshape([43, 384, 14, 14])
        var_6 = paddle.nn.functional.conv._conv_nd(var_5, self.parameter_6, bias=self.parameter_0, stride=[2, 2], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_7 = var_6.reshape([43, 384, 49])
        var_8 = var_7.transpose([0, 2, 1])
        var_9 = paddle.nn.functional.norm.layer_norm(var_8, normalized_shape=[384], weight=self.parameter_2, bias=self.parameter_1, epsilon=1e-05)
        var_10 = paddle.nn.functional.common.linear(x=var_9, weight=self.parameter_3, bias=self.parameter_4, name=None)
        var_11 = var_10.reshape([43, 49, 2, 12, 32])
        var_12 = var_11.transpose([2, 0, 3, 1, 4])
        var_13 = var_12.__getitem__(0)
        var_14 = var_12.__getitem__(1)
        var_15 = var_13.transpose([0, 1, 3, 2])
        return var_3, var_15, var_14



# def create_inputspec(): 
#     inputspec = ( 
#         paddle.static.InputSpec(shape=(-1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
#     )
#     return inputspec

# def create_tensor_inputs():
#     inputs = (
#         paddle.rand(shape=[43, 196, 384], dtype=paddle.float32),
#     )
#     return inputs


# def create_numpy_inputs():
#     inputs = (
#         np.random.random(size=[43, 196, 384]).astype('float32'),
#     )
#     return inputs

def trimmean(data_list, ratio=0.2):
    """
    掐头去尾求平均
    :param data_list: 输入的data list, 多次试验的结果集合
    """
    head = int(len(data_list) * ratio)
    tail = int(len(data_list) - len(data_list) * ratio)
    res = sum(sorted(data_list)[head:tail]) / (tail - head)
    return res

def dy_eval_perf():
    """dygraph eval"""
    net = LayerCase()
    st_net = paddle.jit.to_static(net, full_graph=True)
    net.eval()

    def _perf(input_data):
        logit = net(*input_data)
        return logit
    total_time_list = []

    data = [
        paddle.to_tensor(np.random.random(size=[43, 196, 384]).astype('float32'), stop_gradient=False)
    ]

    # 预热
    timeit.timeit(lambda: _perf(data), number=10)
    # timeit.timeit(lambda: _perf(data), number=int(self.perf_repeat * 1 * 0.2))
    for i in range(1000):
        total_time = timeit.timeit(lambda: _perf(data), number=1)
        total_time_list.append(total_time)
    
    time_res = trimmean(data_list=total_time_list)
    time_res = round(time_res * 100, 6)
    return time_res

for i in range(300):
    res = dy_eval_perf()
    print(f'step {i}: {res}')
