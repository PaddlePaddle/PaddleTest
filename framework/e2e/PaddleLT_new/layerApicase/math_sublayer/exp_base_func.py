import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: exp_base
    api简介: 对输入, 逐元素进行以自然数e为底指数运算
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.exp(x,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-2 + (2 - -2) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-2 + (2 - -2) * np.random.random([2, 3, 4, 4]).astype('float32'), )
    return inputs

