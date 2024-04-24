import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: nonzero_0
    api简介: 返回输入 x 中非零元素的坐标
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.nonzero(x,  as_tuple=False, )
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
    inputs = (paddle.to_tensor([[1.0, 1.0, 4.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype='float32', stop_gradient=False), )
    return inputs

