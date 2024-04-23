import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: SpectralNorm_9
    api简介: 谱归一化
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.SpectralNorm(weight_shape=[2, 3, 8, 8], dim=1, power_iters=1, eps=1e-12, dtype='float32', )

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
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 8, 8]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 8, 8]).astype('float32'), )
    return inputs

