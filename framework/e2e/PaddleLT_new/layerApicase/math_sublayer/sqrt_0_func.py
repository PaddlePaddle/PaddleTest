import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: sqrt_0
    api简介: 计算输入的算数平方根
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.sqrt(x,  )
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
    inputs = (paddle.to_tensor([0, 0, 0, 0], dtype='float32', stop_gradient=False), )
    return inputs

