import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: broadcast_to_0
    api简介: 根据 shape 指定的形状广播 x ，广播后， x 的形状和 shape 指定的形状一致
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.broadcast_to(x,  shape=[1, 2, 1, 3], )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([1, 3]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([1, 3]).astype('float32'), )
    return inputs

