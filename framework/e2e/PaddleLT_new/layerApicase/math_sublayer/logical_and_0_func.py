import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: logical_and_0
    api简介: 逐元素的对 x 和 y 进行逻辑与运算
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """
        out = paddle.logical_and(x, y,  )
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
    inputs = (paddle.to_tensor([True], dtype='bool', stop_gradient=False), paddle.to_tensor([True, False, True, False], dtype='bool', stop_gradient=False), )
    return inputs

