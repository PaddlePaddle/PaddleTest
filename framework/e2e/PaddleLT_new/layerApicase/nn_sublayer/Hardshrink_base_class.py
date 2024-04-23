import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Hardshrink_base
    api简介: Hardshrink激活层
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Hardshrink()

    def forward(self, data, ):
        """
        forward
        """
        out = self.func(data, )
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
    inputs = (paddle.to_tensor([-1, 0.3, 2.5], dtype='float32', stop_gradient=False), )
    return inputs

