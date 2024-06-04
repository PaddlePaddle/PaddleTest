import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: tensordot_4
    api简介: 张量缩并运算（Tensor Contraction），即沿着axes给定的多个轴对两个张量对应元素的乘积进行加和操作
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """
        out = paddle.tensordot(x, y,  axes=([0, 1], [1, 0]), )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 4, 5, 6]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-1 + (1 - -1) * np.random.random([3, 2, 4, 5, 6]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 4, 5, 6]).astype('float32'), -1 + (1 - -1) * np.random.random([3, 2, 4, 5, 6]).astype('float32'), )
    return inputs

