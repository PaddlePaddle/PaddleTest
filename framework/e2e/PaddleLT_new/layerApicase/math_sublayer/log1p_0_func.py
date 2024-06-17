import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: log1p_0
    api简介: 计算Log1p（加一的自然对数）结果
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.log1p(x,  )
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
    inputs = (paddle.to_tensor(0.5 + (20 - 0.5) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (0.5 + (20 - 0.5) * np.random.random([2, 3, 4, 4]).astype('float32'), )
    return inputs

