import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: inner_0
    api简介: 计算两个Tensor的内积
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.inner(x,  y=paddle.to_tensor(-10 + (10 - -10) * np.random.random([1, 3, 2, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([2, 1, 5, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([2, 1, 5, 4]).astype('float32'), )
    return inputs

