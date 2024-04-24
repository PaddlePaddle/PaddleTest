import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: kron_3
    api简介: 计算两个张量的克罗内克积, 结果是一个合成的张量, 由第二个张量经过第一个张量中的元素缩放 后的组块构成。
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """
        out = paddle.kron(x, y,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-10 + (10 - -10) * np.random.random([2, 3, 5, 5]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([2, 3, 4, 4]).astype('float32'), -10 + (10 - -10) * np.random.random([2, 3, 5, 5]).astype('float32'), )
    return inputs

