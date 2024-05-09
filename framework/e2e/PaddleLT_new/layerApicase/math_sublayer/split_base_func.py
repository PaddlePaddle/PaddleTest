import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: split_base
    api简介: 该OP将输入Tensor分割成多个子Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.split(x,  num_or_sections=[2, -1, 1], axis=0, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([4, 6, 6]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([4, 6, 6]).astype('float32'), )
    return inputs

