import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: addmm_5
    api简介: 计算x和y的乘积，将结果乘以标量alpha，再加上input与beta的乘积，得到输出
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, input, x, y, ):
        """
        forward
        """
        out = paddle.addmm(input, x, y,  alpha=1, beta=1, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([5, 5]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-10 + (10 - -10) * np.random.random([5, 3]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-10 + (10 - -10) * np.random.random([3, 5]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([5, 5]).astype('float32'), -10 + (10 - -10) * np.random.random([5, 3]).astype('float32'), -10 + (10 - -10) * np.random.random([3, 5]).astype('float32'), )
    return inputs

