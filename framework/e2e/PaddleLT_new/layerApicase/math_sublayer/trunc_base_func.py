import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: trunc_base
    api简介: 将输入 Tensor 的小数部分置0，返回置0后的 Tensor ，如果输入 Tensor 的数据类型为整数，则不做处理
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, input, ):
        """
        forward
        """
        out = paddle.trunc(input,  )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([2, 3, 4, 4]).astype('float32'), )
    return inputs

