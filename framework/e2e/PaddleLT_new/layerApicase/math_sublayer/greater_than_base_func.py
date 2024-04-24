import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: greater_than_base
    api简介: 逐元素地返回 x>y 的逻辑值，相同位置前者输入大于后者输入则返回True，否则返回False
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """
        out = paddle.greater_than(x, y,  )
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
    inputs = (paddle.to_tensor([[2.0, 1.0, -3.5], [-2.7, 1.5, 3], [0, 4.1, 8.6]], dtype='float32', stop_gradient=False), paddle.to_tensor([[-2.0, 1.1, -3.5], [-2.5, 1.5, 3.5], [0.5, 4.2, 8.3]], dtype='float32', stop_gradient=False), )
    return inputs

