import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: unique_0
    api简介: 返回Tensor按升序排序后的独有元素
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.unique(x,  axis=1, )
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
    inputs = (paddle.to_tensor([[0.4, 0.4], [0.7, 0.9]], dtype='float32', stop_gradient=False), )
    return inputs

