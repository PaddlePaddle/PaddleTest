import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Swish_0
    api简介: Swish激活层
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Swish()

    def forward(self, data, ):
        """
        forward
        """
        out = self.func(data, )
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
    inputs = (paddle.to_tensor(-4 + (4 - -4) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-4 + (4 - -4) * np.random.random([2, 3, 4, 4]).astype('float32'), )
    return inputs

