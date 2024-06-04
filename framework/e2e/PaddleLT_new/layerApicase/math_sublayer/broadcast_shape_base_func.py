import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: broadcast_shape_base
    api简介: 该函数返回对x_shape大小的张量和y_shape大小的张量做broadcast操作后得到的shape
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """
        out = paddle.broadcast_shape( x_shape=[2, 3, 4, 4, 1], y_shape=[2, 3, 1, 4, 3], )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = ()
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = ()
    return inputs

