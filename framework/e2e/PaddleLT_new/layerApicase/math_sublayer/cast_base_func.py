import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: cast_base
    api简介: 将 x 的数据类型转换为 dtype 并输出。支持输出和输入的数据类型相同
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.cast(x,  dtype='int32', )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([2, 3, 4, 4]).astype('float32'), )
    return inputs

