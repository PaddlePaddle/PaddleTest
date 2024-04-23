import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: conv1d_base
    api简介: 1维卷积
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.nn.functional.conv1d(x,  weight=paddle.to_tensor(-1 + (1 - -1) * np.random.random([1, 3, 3]).astype('float32'), dtype='float32', stop_gradient=False), bias=paddle.to_tensor(-1 + (1 - -1) * np.random.random([1]).astype('float32'), dtype='float32', stop_gradient=False), stride=1, padding=0, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 4]).astype('float32'), )
    return inputs

