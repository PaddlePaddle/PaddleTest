import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: conj_0
    api简介: 是逐元素计算Tensor的共轭运算
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.conj(x,  )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.complex64, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor((-2 + (2 - -2) * np.random.random([2, 3, 4, 4]) + (-2 + (2 - -2) * np.random.random([2, 3, 4, 4])) * 1j).astype(np.complex64), dtype='complex64', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = ((-2 + (2 - -2) * np.random.random([2, 3, 4, 4]) + (-2 + (2 - -2) * np.random.random([2, 3, 4, 4])) * 1j).astype(np.complex64), )
    return inputs

