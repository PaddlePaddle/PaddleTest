import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: broadcast_to_base
    api简介: 根据 shape 指定的形状广播 x ，广播后， x 的形状和 shape 指定的形状一致
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.broadcast_to(x,  shape=[2, 3, 4, 3, 5], )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 1, 3, 1]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 1, 3, 1]).astype('float32'), )
    return inputs

