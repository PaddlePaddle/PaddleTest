import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Pad1D_23
    api简介: 1维pad填充
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Pad1D(padding=[1, 2], mode='replicate', data_format='NLC', )

    def forward(self, data, ):
        """
        forward
        """
        out = self.func(data, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(1, 2, 3), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([1, 2, 3]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([1, 2, 3]).astype('float32'), )
    return inputs

