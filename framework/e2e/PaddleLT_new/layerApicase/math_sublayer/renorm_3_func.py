import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: renorm_3
    api简介: 求Tensor的renorm值
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.renorm(x,  p=1.5, axis=2, max_norm=20, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[10.0, 10.0, 10.0], [0.0, 0.0, 0.0]], [[6.0, 6.0, 6.0], [3.0, 3.0, 3.0]]], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[10.0, 10.0, 10.0], [0.0, 0.0, 0.0]], [[6.0, 6.0, 6.0], [3.0, 3.0, 3.0]]]).astype('float32'), )
    return inputs

