import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Dropout2D_2
    api简介: 2维Dropout
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Dropout2D(p=0.5, data_format='NCHW', )

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
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 4, 5]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 4, 5]).astype('float32'), )
    return inputs

