import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: var_1
    api简介: 沿给定的轴 axis 计算 x 中元素的方差
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.var(x,  axis=2, unbiased=False, keepdim=False, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(2, 3, 4, 4), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

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

