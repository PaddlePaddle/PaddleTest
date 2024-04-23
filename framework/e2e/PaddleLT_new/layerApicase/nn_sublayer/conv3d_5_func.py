import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: conv3d_5
    api简介: 3维卷积
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.nn.functional.conv3d(x,  weight=paddle.to_tensor(-1 + (1 - -1) * np.random.random([6, 1, 3, 3, 3]).astype('float32'), dtype='float32', stop_gradient=False), bias=paddle.to_tensor(-1 + (1 - -1) * np.random.random([6]).astype('float32'), dtype='float32', stop_gradient=False), stride=2, padding=0, groups=3, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 4, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 4, 4, 4]).astype('float32'), )
    return inputs

