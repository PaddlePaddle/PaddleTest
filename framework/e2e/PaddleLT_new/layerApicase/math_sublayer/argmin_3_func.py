import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: argmin_3
    api简介: 沿参数``axis`` 计算输入 x 的最小元素的索引
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.argmin(x,  axis=-1, keepdim=True, )
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
    inputs = (paddle.to_tensor([0, 1, 2], dtype='float32', stop_gradient=False), )
    return inputs

