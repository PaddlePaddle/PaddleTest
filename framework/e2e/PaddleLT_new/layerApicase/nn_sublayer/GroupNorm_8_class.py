import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: GroupNorm_8
    api简介: 分组归一化
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.GroupNorm(num_groups=2, num_channels=6, epsilon=1e-05, data_format='NCHW', )

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
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([4, 6, 8, 8]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([4, 6, 8, 8]).astype('float32'), )
    return inputs

