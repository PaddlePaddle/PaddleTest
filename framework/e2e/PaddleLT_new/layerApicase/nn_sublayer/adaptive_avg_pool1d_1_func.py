import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: adaptive_avg_pool1d_1
    api简介: 1维自适应平均池化
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.nn.functional.adaptive_avg_pool1d(x,  output_size=8, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([4, 3, 16]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([4, 3, 16]).astype('float32'), )
    return inputs

