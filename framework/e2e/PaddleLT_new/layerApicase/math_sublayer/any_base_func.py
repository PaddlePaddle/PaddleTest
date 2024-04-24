import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: any_base
    api简介: 对指定维度上的Tensor元素进行逻辑或运算，并输出相应的计算结果
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.any(x,  )
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
    inputs = (paddle.to_tensor([[True, False], [False, True]], dtype='bool', stop_gradient=False), )
    return inputs

