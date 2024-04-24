import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: squeeze__0
    api简介: 删除输入Tensor的Shape中尺寸为1的维度, inplace策略
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.squeeze_(x,  axis=-4, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([1, 2, 1, 3]).astype('float32'), dtype='float32', stop_gradient=True), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([1, 2, 1, 3]).astype('float32'), )
    return inputs

