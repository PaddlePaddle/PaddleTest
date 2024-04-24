import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: put_along_axis_base
    api简介: 基于输入index矩阵, 将输入value沿着指定axis放置入arr矩阵
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, arr, ):
        """
        forward
        """
        out = paddle.put_along_axis(arr,  indices=paddle.to_tensor([[[[0]]]], dtype='int32', stop_gradient=False), value=21.0, axis=0, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), )
    return inputs

