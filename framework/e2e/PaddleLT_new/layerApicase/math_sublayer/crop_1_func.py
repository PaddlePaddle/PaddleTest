import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: crop_1
    api简介: 根据偏移量（offsets）和形状（shape），裁剪输入（x）Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.crop(x,  shape=[2, 1, -1, 2], offsets=[0, 0, 1, 1], )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([2, 3, 3, 3]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([2, 3, 3, 3]).astype('float32'), )
    return inputs

