import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: matmul_1
    api简介: 计算两个Tensor的乘积，遵循完整的广播规则
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """
        out = paddle.matmul(x, y,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([10, 1, 5, 2]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-1 + (1 - -1) * np.random.random([1, 3, 2, 5]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([10, 1, 5, 2]).astype('float32'), -1 + (1 - -1) * np.random.random([1, 3, 2, 5]).astype('float32'), )
    return inputs

