import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: eye_1
    api简介: 创建形状大小为shape并且数据类型为dtype的Tensor，其中元素值是未初始化的
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """
        out = paddle.eye( num_rows=5, num_columns=3, dtype='int64', )
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

