import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: max_pool3d_1
    api简介: 3维最大池化
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.nn.functional.max_pool3d(x,  kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], ceil_mode=True, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 8, 8, 8]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 8, 8, 8]).astype('float32'), )
    return inputs

