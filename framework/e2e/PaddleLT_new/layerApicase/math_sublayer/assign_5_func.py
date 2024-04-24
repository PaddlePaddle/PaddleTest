import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: assign_5
    api简介: 将输入Tensor或numpy数组拷贝至输出Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """
        out = paddle.assign( x=(2, 3, 4, 4), output=paddle.to_tensor(-2 + (1 - -2) * np.random.random([4]).astype('float32'), dtype='float32', stop_gradient=False), )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = ()
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = ()
    return inputs

