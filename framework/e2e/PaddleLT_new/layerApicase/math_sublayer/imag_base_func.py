import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: imag_base
    api简介: 返回一个包含输入复数Tensor的虚部数值的新Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.imag(x,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([2, 3, 4, 4]) + (-10 + (10 - -10) * np.random.random([2, 3, 4, 4])) * 1j).astype(np.complex64), dtype='complex64', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = ((-10 + (10 - -10) * np.random.random([2, 3, 4, 4]) + (-10 + (10 - -10) * np.random.random([2, 3, 4, 4])) * 1j).astype(np.complex64), )
    return inputs

