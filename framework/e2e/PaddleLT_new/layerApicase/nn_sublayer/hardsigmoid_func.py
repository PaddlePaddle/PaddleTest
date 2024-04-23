import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: hardsigmoid
    api简介: hardsigmoid激活函数
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.nn.functional.hardsigmoid(x,  slope=0.1666667, offset=0.5, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-2 + (5 - -2) * np.random.random([2, 3, 8, 8]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-2 + (5 - -2) * np.random.random([2, 3, 8, 8]).astype('float32'), )
    return inputs

