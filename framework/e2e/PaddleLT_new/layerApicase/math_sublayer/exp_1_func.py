import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: exp_1
    api简介: AD测试：init=-1
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
    inputs = (paddle.to_tensor(-2 + (2 - -2) * np.random.random([4, 4]).astype('float64'), dtype='float64', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-2 + (2 - -2) * np.random.random([4, 4]).astype('float64'), )
    return inputs

