import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: not_equal_2
    api简介: 返回 x!=y 逐元素比较x和y是否相等，相同位置的元素不相同则返回True，否则返回False
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """
        out = paddle.not_equal(x, y,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(-1, 1, [1, 2]).astype('int32'), dtype='int32', stop_gradient=False), paddle.to_tensor(np.random.randint(-1, 1, [2, 2]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(-1, 1, [1, 2]).astype('int32'), np.random.randint(-1, 1, [2, 2]).astype('int32'), )
    return inputs

