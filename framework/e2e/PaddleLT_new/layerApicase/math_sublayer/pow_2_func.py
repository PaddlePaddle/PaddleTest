import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: pow_2
    api简介: 指数算子，逐元素计算 x 的 y 次幂指数算子，逐元素计算 x 的 y 次幂
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """
        out = paddle.pow(x, y,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([1, 2, 1, 3]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-2 + (6 - -2) * np.random.random([1, 2, 3]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([1, 2, 1, 3]).astype('float32'), -2 + (6 - -2) * np.random.random([1, 2, 3]).astype('float32'), )
    return inputs

