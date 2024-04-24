import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: log2_0
    api简介: Log2激活函数(计算底为2对数)
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.log2(x,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([0], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([0]).astype('float32'), )
    return inputs

