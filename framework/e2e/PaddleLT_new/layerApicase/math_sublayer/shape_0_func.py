import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: shape_0
    api简介: 获得输入Tensor或SelectedRows的shape
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, input, ):
        """
        forward
        """
        out = paddle.shape(input,  )
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
    inputs = (paddle.to_tensor([[]], dtype='float32', stop_gradient=False), )
    return inputs

