import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: digamma_0
    api简介: 逐元素计算输入Tensor的digamma函数值
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.digamma(x,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-5 + (5 - -5) * np.random.random([3, 6, 6, 6, 6]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-5 + (5 - -5) * np.random.random([3, 6, 6, 6, 6]).astype('float32'), )
    return inputs

