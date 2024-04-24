import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: cumprod_base
    api简介: 沿给定 axis 计算张量 x 的累乘
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.cumprod(x,  dim=0, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-2 + (2 - -2) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-2 + (2 - -2) * np.random.random([2, 3, 4, 4]).astype('float32'), )
    return inputs

