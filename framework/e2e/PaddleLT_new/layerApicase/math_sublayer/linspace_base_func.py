import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: linspace_base
    api简介: 返回一个Tensor，Tensor的值为在区间start和stop上均匀间隔的num个值，输出Tensor的长度为num
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """
        out = paddle.linspace( start=-3.0, stop=15.0, num=4, dtype='float32', )
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

