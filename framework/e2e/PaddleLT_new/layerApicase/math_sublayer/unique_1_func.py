import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: unique_1
    api简介: 返回Tensor按升序排序后的独有元素
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.unique(x,  axis=0, )
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
    inputs = (paddle.to_tensor([[8, 4], [7, 9]], dtype='int32', stop_gradient=False), )
    return inputs

