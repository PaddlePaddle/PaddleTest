import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: equal_all_1
    api简介: 如果所有相同位置的元素相同返回True，否则返回False
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """
        out = paddle.equal_all(x, y,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([3, 3, 3]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-10 + (10 - -10) * np.random.random([3, 3, 3]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([3, 3, 3]).astype('float32'), -10 + (10 - -10) * np.random.random([3, 3, 3]).astype('float32'), )
    return inputs

