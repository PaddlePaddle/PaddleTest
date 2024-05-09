import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Conv2DTranspose_2
    api简介: 2维反卷积
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Conv2DTranspose(in_channels=3, out_channels=1, kernel_size=[3, 3], stride=1, padding=[1, 0], dilation=1, )

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
    inputs = (paddle.to_tensor(0 + (1 - 0) * np.random.random([2, 3, 2, 2]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (0 + (1 - 0) * np.random.random([2, 3, 2, 2]).astype('float32'), )
    return inputs

