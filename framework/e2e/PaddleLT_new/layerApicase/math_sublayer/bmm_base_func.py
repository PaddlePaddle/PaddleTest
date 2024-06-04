import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: bmm_base
    api简介: 对输入x及输入y进行矩阵相乘
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """
        out = paddle.bmm(x, y,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 5, 4]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 4, 5]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 5, 4]).astype('float32'), -1 + (1 - -1) * np.random.random([2, 4, 5]).astype('float32'), )
    return inputs

