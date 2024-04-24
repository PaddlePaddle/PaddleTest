import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: equal_3
    api简介: 逐元素比较x和y是否相等，相同位置的元素相同则返回True，否则返回False
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """
        out = paddle.equal(x, y,  )
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
    inputs = (paddle.to_tensor([[[[1, 2, 3], [1, 2, 3], [1, 2, 3]]]], dtype='float32', stop_gradient=False), paddle.to_tensor([[1, 2, 3]], dtype='float32', stop_gradient=False), )
    return inputs

