import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: floor_divide_0
    api简介: 输入 x 与输入 y 逐元素整除，并将各个位置的输出元素保存到返回结果中
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """
        out = paddle.floor_divide(x, y,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(1, 10, [3, 3, 3]).astype('int32'), dtype='int32', stop_gradient=False), paddle.to_tensor(np.random.randint(1, 10, [3]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(1, 10, [3, 3, 3]).astype('int32'), np.random.randint(1, 10, [3]).astype('int32'), )
    return inputs

