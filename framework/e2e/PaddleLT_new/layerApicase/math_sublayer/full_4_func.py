import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: full_4
    api简介: 创建形状大小为 shape 并且数据类型为 dtype 的Tensor，其中元素值均为 fill_value
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """
        out = paddle.full( shape=paddle.to_tensor([2, 3, 4, 4], dtype='int32', stop_gradient=False), fill_value=1, dtype='int32', )
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
    inputs = ()
    return inputs

