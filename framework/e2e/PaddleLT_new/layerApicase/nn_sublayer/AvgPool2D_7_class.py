import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: AvgPool2D_7
    api简介: 2维平均池化
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.AvgPool2D(kernel_size=[3, 3], stride=[3, 3], padding=[0, 0, 0, 0], )

    def forward(self, data, ):
        """
        forward
        """
        out = self.func(data, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([2, 3, 32, 32]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([2, 3, 32, 32]).astype('float32'), )
    return inputs

