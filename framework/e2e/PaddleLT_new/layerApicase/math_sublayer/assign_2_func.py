import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: assign_2
    api简介: 将输入Tensor或numpy数组拷贝至输出Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.assign(x,  output=paddle.to_tensor(np.random.randint(0, 2, [2, 3, 4, 4]).astype('bool'), dtype='bool', stop_gradient=False), )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (2 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (2 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), )
    return inputs

