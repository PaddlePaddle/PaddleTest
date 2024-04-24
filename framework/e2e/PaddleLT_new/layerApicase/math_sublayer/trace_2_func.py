import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: trace_2
    api简介: 计算输入 Tensor 在指定平面上的对角线元素之和，并输出相应的计算结果
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.trace(x,  offset=1, axis1=0, axis2=1, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([3, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([3, 4]).astype('float32'), )
    return inputs

